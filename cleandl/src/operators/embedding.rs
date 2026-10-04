use crate::{
    context::{Context, ResourceKey},
    include_cubins,
    tensor::{Tensor, TensorView, TensorViewMut},
};
use skycuda::{module::Module, stream::Stream};
use skytens_core::{
    data_type::{ElementType, FloatType, UIntType},
    shape::Shape,
};
use std::sync::Arc;

#[derive(Debug)]
pub struct EmbeddingExecutor {
    cx: Context,
    module: Arc<Module>,
}

impl EmbeddingExecutor {
    pub fn get(cx: &Context) -> anyhow::Result<Arc<Self>> {
        cx.get_or_create_resource(EmbeddingExecutorKey)
    }

    /// Returns the looked-up embeddings as bf16 `[num_tokens, embed_dim]`.
    pub fn embedding_lookup(
        &self,
        embedding_table: &TensorView<'_>,
        token_indices: &TensorView<'_>,
        stream: &Stream,
    ) -> anyhow::Result<Tensor> {
        assert_eq!(
            embedding_table.element_type(),
            ElementType::Float(FloatType::F32)
        );
        assert_eq!(
            token_indices.element_type(),
            ElementType::UInt(UIntType::U32)
        );

        assert!(embedding_table.is_packed_row_major());
        assert!(token_indices.is_packed_row_major());

        let &[vocab_size, embed_dim] = embedding_table.shape().dims_slice() else {
            panic!("wrong shape for embedding table");
        };
        assert!(embed_dim.is_multiple_of(128));
        let &[num_tokens] = token_indices.shape().dims_slice() else {
            panic!("wrong shape for token indices")
        };

        let mut out_embeddings = Tensor::new_zeroed(
            Shape::new([num_tokens, embed_dim]),
            ElementType::Float(FloatType::Bf16),
            &self.cx,
        )?;

        let kernel = self
            .module
            .get_function("embedding_lookup_f32in_bf16out_32x128")?;

        {
            let token_indices_memory = token_indices.storage();
            let embedding_table_memory = embedding_table.storage();
            let mut launch = stream.launch_builder(&kernel);
            launch
                .arg(token_indices_memory)?
                .arg(embedding_table_memory)?
                .arg(out_embeddings.storage_mut())?
                .arg(i32::try_from(vocab_size)?)?
                .arg(i32::try_from(num_tokens)?)?
                .arg(i32::try_from(embed_dim)?)?;
            // Block dim and smem are overriden by TileIR.
            unsafe {
                launch.launch(
                    embed_dim.div_ceil(128) as u32,
                    num_tokens.div_ceil(32) as u32,
                    1,
                )?;
            }
        }

        Ok(out_embeddings)
    }

    /// Accumulates the embedding-table gradient in-place; `dl_din` must be pre-zeroed.
    pub fn embedding_scatter_add(
        &self,
        token_indices: &TensorView<'_>,
        dl_dout: &TensorView<'_>,
        dl_din: &mut TensorViewMut<'_>,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        assert_eq!(
            token_indices.element_type(),
            ElementType::UInt(UIntType::U32)
        );
        assert_eq!(dl_dout.element_type(), ElementType::Float(FloatType::Bf16));
        assert_eq!(dl_din.element_type(), ElementType::Float(FloatType::F32));

        assert!(dl_dout.is_packed_row_major());
        assert!(token_indices.is_packed_row_major());
        assert!(dl_din.is_packed_row_major());

        let &[vocab_size, embed_dim] = dl_din.shape().dims_slice() else {
            panic!("wrong shape for gradient embedding table");
        };
        assert!(embed_dim.is_multiple_of(128));
        let &[num_tokens] = token_indices.shape().dims_slice() else {
            panic!("wrong shape for token indices")
        };
        assert_eq!(dl_dout.shape(), Shape::new([num_tokens, embed_dim]));

        let kernel = self
            .module
            .get_function("embedding_scatter_add_bf16in_f32out_32x128")?;

        let token_indices_memory = token_indices.storage();
        let dl_dout_memory = dl_dout.storage();
        let mut dl_din_memory = dl_din.storage_mut();
        let mut launch = stream.launch_builder(&kernel);
        launch
            .arg(token_indices_memory)?
            .arg(dl_dout_memory)?
            .arg(&mut dl_din_memory)?
            .arg(i32::try_from(vocab_size)?)?
            .arg(i32::try_from(num_tokens)?)?
            .arg(i32::try_from(embed_dim)?)?;
        // Block dim and smem are overriden by TileIR.
        unsafe {
            launch.launch(
                embed_dim.div_ceil(128) as u32,
                num_tokens.div_ceil(32) as u32,
                1,
            )?;
        }
        Ok(())
    }
}

#[derive(Debug, serde::Serialize)]
struct EmbeddingExecutorKey;

impl ResourceKey for EmbeddingExecutorKey {
    type Resource = EmbeddingExecutor;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        Ok(EmbeddingExecutor {
            cx: cx.clone(),
            module: cx.get_or_create_resource(include_cubins!("../../cutile-kernels/embedding"))?,
        })
    }
}
