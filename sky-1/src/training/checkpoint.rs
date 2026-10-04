use crate::training::parameter::Parameter;
use anyhow::{Context as _, ensure};
use cleandl::{context::Context, tensor::Tensor};
use safetensors::{Dtype, SafeTensors, tensor::Metadata};
use skycuda::{host_pinned_memory::HostPinnedMemory, stream::Stream};
use skytens_core::{
    data_type::{ElementType, FloatType},
    shape::Shape,
};
use std::path::Path;

#[derive(Debug, Default)]
pub struct CheckpointBuilder {
    prefix_stack: Vec<String>,
    model_tensors: Vec<SerializableTensor>,
    optimizer_tensors: Vec<SerializableTensor>,
}

impl CheckpointBuilder {
    pub fn add_param(
        &mut self,
        name: &str,
        param: &Parameter,
        cx: &Context,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        let name = format!("{}.{}", self.current_prefix(), name);

        let weight_data = cx.download_from_device(param.master_weights.storage(), stream)?;
        let adam_m_data = cx.download_from_device(param.adam_m.storage(), stream)?;
        let adam_v_data = cx.download_from_device(&param.adam_v.storage(), stream)?;

        self.model_tensors.push(SerializableTensor {
            name: name.clone(),
            shape: param.shape(),
            data: weight_data,
            safetensors_dtype: Dtype::F32,
        });
        self.optimizer_tensors.push(SerializableTensor {
            name: format!("{name}:adam_m"),
            shape: param.shape(),
            data: adam_m_data,
            safetensors_dtype: Dtype::F32,
        });
        self.optimizer_tensors.push(SerializableTensor {
            name: format!("{name}:adam_v"),
            shape: param.shape(),
            data: adam_v_data,
            safetensors_dtype: Dtype::F32,
        });

        Ok(())
    }

    pub fn build_child(
        &mut self,
        id: &str,
        build_child: impl FnOnce(&mut Self) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        self.prefix_stack.push(id.to_owned());
        let res = build_child(self);
        self.prefix_stack.pop();
        res
    }

    fn current_prefix(&self) -> String {
        self.prefix_stack.join("/")
    }

    pub fn save_to_dir(&mut self, train_step: u32, dir: impl AsRef<Path>) -> anyhow::Result<()> {
        let dir = dir.as_ref();

        fn make_safetensor<'a>(
            t: &'a mut SerializableTensor,
        ) -> anyhow::Result<(&'a str, safetensors::tensor::TensorView<'a>)> {
            let data = t.data.view()?;
            let data = safetensors::tensor::TensorView::new(
                t.safetensors_dtype,
                t.shape.dims_slice().to_vec(),
                data,
            )?;

            Ok((t.name.as_str(), data))
        }

        let model_tensors: Vec<_> = self
            .model_tensors
            .iter_mut()
            .map(make_safetensor)
            .collect::<anyhow::Result<_>>()?;
        safetensors::tensor::serialize_to_file(
            model_tensors,
            None,
            &dir.join("model.safetensors"),
        )?;

        let optimizer_tensors: Vec<_> = self
            .optimizer_tensors
            .iter_mut()
            .map(make_safetensor)
            .collect::<anyhow::Result<_>>()?;
        let metadata = [("train_step".to_owned(), train_step.to_string())]
            .into_iter()
            .collect();
        safetensors::tensor::serialize_to_file(
            optimizer_tensors,
            Some(metadata),
            &dir.join("optimizer.safetensors"),
        )?;

        Ok(())
    }
}

#[derive(Debug)]
struct SerializableTensor {
    name: String,
    shape: Shape,
    data: HostPinnedMemory,
    safetensors_dtype: Dtype,
}

#[derive(Debug)]
pub struct CheckpointLoader {
    prefix_stack: Vec<String>,
    model_file: SafetensorsFile,
    optimizer_file: SafetensorsFile,
    train_step: u32,
}

impl CheckpointLoader {
    pub fn load_from_dir(dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let dir = dir.as_ref();

        let model_file = SafetensorsFile::read(&dir.join("model.safetensors"))?;
        let optimizer_file = SafetensorsFile::read(&dir.join("optimizer.safetensors"))?;

        let train_step = optimizer_file
            .metadata
            .metadata()
            .as_ref()
            .and_then(|m| m.get("train_step"))
            .context("optimizer checkpoint is missing train_step metadata")?
            .parse()?;

        Ok(Self {
            prefix_stack: Vec::new(),
            model_file,
            optimizer_file,
            train_step,
        })
    }

    pub fn train_step(&self) -> u32 {
        self.train_step
    }

    pub fn load_param(
        &mut self,
        name: &str,
        cx: &Context,
        stream: &Stream,
    ) -> anyhow::Result<Parameter> {
        let name = format!("{}.{}", self.current_prefix(), name);

        let master_weights = self.model_file.load_f32_tensor(&name, cx, stream)?;
        let adam_m = self
            .optimizer_file
            .load_f32_tensor(&format!("{name}:adam_m"), cx, stream)?;
        let adam_v = self
            .optimizer_file
            .load_f32_tensor(&format!("{name}:adam_v"), cx, stream)?;

        ensure!(
            adam_m.shape() == master_weights.shape() && adam_v.shape() == master_weights.shape(),
            "optimizer state shape mismatch for parameter '{name}'"
        );

        Parameter::with_initial_weights_and_opt_state(master_weights, adam_m, adam_v, stream, cx)
    }

    pub fn load_child<T>(
        &mut self,
        id: &str,
        load_child: impl FnOnce(&mut Self) -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        self.prefix_stack.push(id.to_owned());
        let res = load_child(self);
        self.prefix_stack.pop();
        res
    }

    fn current_prefix(&self) -> String {
        self.prefix_stack.join("/")
    }
}

#[derive(Debug)]
struct SafetensorsFile {
    buffer: Vec<u8>,
    header_len: usize,
    metadata: Metadata,
}

impl SafetensorsFile {
    fn read(path: &Path) -> anyhow::Result<Self> {
        let buffer = fs_err::read(path)?;
        let (header_len, metadata) = SafeTensors::read_metadata(&buffer)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        Ok(Self {
            buffer,
            header_len,
            metadata,
        })
    }

    fn load_f32_tensor(&self, name: &str, cx: &Context, stream: &Stream) -> anyhow::Result<Tensor> {
        let info = self
            .metadata
            .info(name)
            .with_context(|| format!("checkpoint is missing tensor '{name}'"))?;
        ensure!(
            info.dtype == Dtype::F32,
            "expected tensor '{name}' to have dtype F32, found {:?}",
            info.dtype
        );

        let data_start = 8 + self.header_len;
        let (start, end) = info.data_offsets;
        let bytes = &self.buffer[data_start + start..data_start + end];

        Tensor::from_bytes(
            bytes,
            ElementType::Float(FloatType::F32),
            Shape::new(info.shape.iter().copied()),
            stream,
            cx,
        )
    }
}
