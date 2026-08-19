use std::sync::Arc;

use singe_cuda::{
    dim::Dim3,
    module::{KernelParameters, LaunchConfig, Module},
};
use skytens_core::{
    data_type::{ElementType, FloatType, UIntType},
    shape::Shape,
};

use crate::{
    context::{Context, CudaStream, ResourceKey},
    cutile::CutileKernelKey,
    tensor::TensorView,
};

#[derive(Debug)]
pub struct EmbeddingExecutor {
    module: Arc<Module>,
}

impl EmbeddingExecutor {
    pub fn get(cx: &Context) -> anyhow::Result<Arc<Self>> {
        cx.get_or_create_resource(EmbeddingExecutorKey)
    }

    pub fn embedding_lookup(
        &self,
        embedding_table: &TensorView,
        token_indices: &TensorView,
        out_embeddings: &TensorView,
        stream: &CudaStream,
    ) -> anyhow::Result<()> {
        assert_eq!(
            embedding_table.element_type(),
            ElementType::Float(FloatType::F32)
        );
        assert_eq!(
            token_indices.element_type(),
            ElementType::UInt(UIntType::U32)
        );
        assert_eq!(
            out_embeddings.element_type(),
            ElementType::Float(FloatType::Bf16)
        );

        assert!(embedding_table.is_packed_row_major());
        assert!(token_indices.is_packed_row_major());
        assert!(out_embeddings.is_packed_row_major());

        let &[vocab_size, embed_dim] = embedding_table.shape().dims_slice() else {
            panic!("wrong shape for embedding table");
        };
        assert!(embed_dim.is_multiple_of(128));
        let &[num_tokens] = token_indices.shape().dims_slice() else {
            panic!("wrong shape for token indices")
        };
        assert_eq!(out_embeddings.shape(), Shape::new([num_tokens, embed_dim]));

        let kernel = self
            .module
            .function("embedding_lookup_f32in_bf16out_32x128")?;

        kernel.launch_on(
            &LaunchConfig::new(
                Dim3::new(
                    embed_dim.div_ceil(128) as u32,
                    num_tokens.div_ceil(32) as u32,
                    1,
                ),
                Dim3::new(1, 1, 1), // overriden by TileIR
                0,                  // overriden by TileIR
            )?,
            KernelParameters::new()
                .push(token_indices.data_view())
                .push(embedding_table.data_view())
                .push(out_embeddings.data_view())
                .push(i32::try_from(vocab_size)?)
                .push(i32::try_from(num_tokens)?)
                .push(i32::try_from(embed_dim)?),
            stream.get(),
        )?;
        Ok(())
    }

    pub fn embedding_scatter_add(
        &self,
        token_indices: &TensorView,
        dl_dout: &TensorView,
        dl_din: &TensorView,
        stream: &CudaStream,
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

        // Since we use atomic adds to accumulate the gradients, we need to pre-memset
        // dl_din to zero.
        dl_din.memset_u8(0, stream)?;

        let kernel = self
            .module
            .function("embedding_scatter_add_bf16in_f32out_32x128")?;

        kernel.launch_on(
            &LaunchConfig::new(
                Dim3::new(
                    embed_dim.div_ceil(128) as u32,
                    num_tokens.div_ceil(32) as u32,
                    1,
                ),
                Dim3::new(1, 1, 1), // overriden by TileIR
                0,                  // overriden by TileIR
            )?,
            KernelParameters::new()
                .push(token_indices.data_view())
                .push(dl_dout.data_view())
                .push(dl_din.data_view())
                .push(i32::try_from(vocab_size)?)
                .push(i32::try_from(num_tokens)?)
                .push(i32::try_from(embed_dim)?),
            stream.get(),
        )?;
        Ok(())
    }
}

#[derive(Debug, serde::Serialize)]
struct EmbeddingExecutorKey;

impl ResourceKey for EmbeddingExecutorKey {
    type Resource = EmbeddingExecutor;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        Ok(EmbeddingExecutor {
            module: cx.get_or_create_resource(CutileKernelKey {
                src: include_str!("../../cutile-kernels/embedding.cu"),
            })?,
        })
    }
}
