use nvvm_builder::{DType, DynVal, Val, ValueType};

/// Dynamic kernel parameter helper.
pub struct DynParamsBuilder<K> {
    params: Vec<(K, DType)>,
    bound: Option<Vec<DynVal>>,
}

impl<K: Eq> DynParamsBuilder<K> {
    pub fn new() -> Self {
        Self {
            params: Vec::new(),
            bound: None,
        }
    }

    pub fn add(&mut self, key: K, dtype: DType) {
        self.params.push((key, dtype));
    }

    pub fn get_index(&self, key: K) -> usize {
        self.params.iter().position(|(k, _)| k == &key).unwrap()
    }

    pub fn bind(&mut self, vals: Vec<DynVal>) {
        self.bound = Some(vals);
    }

    pub fn get<T: ValueType>(&self, key: K) -> Val<T> {
        self.bound.as_ref().unwrap()[self.get_index(key)]
            .downcast()
            .unwrap()
    }

    pub fn param_dtypes(&self) -> Vec<DType> {
        self.params.iter().map(|(_, dtype)| dtype.clone()).collect()
    }
}

pub fn dtype<T: ValueType<Shape = ()>>() -> DType {
    T::dtype(())
}
