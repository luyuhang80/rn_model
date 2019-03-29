# GCN模块使用说明

GCN模块包含两个类，gcn_specs类用于描述希望构建的图卷积网络，GCN类用于实际的构建

## gcn_specs

这个类应该当作一个结构体使用，它只有成员变量，没有方法

一些比较关键的成员变量：

* n_gconv_layers: 图卷积层的层数，假设为a层，那么接下来的4个变量都应该是含有a个元素的列表
* laplacians：依次是每一个图卷积层的图的拉普帕斯矩阵
* n_gconv_filters：依次是每一个图卷积层的卷积核个数
* polynomial_orders: 依次是每一个图卷积层的傅立叶变换展开数
* pooling_sizes：依次是每一个图卷积层的pooling大小

* fc_dims：图卷积结束后还可以添加全连接层，依次指定每一层的节点数

## GCN

GCN类在初始化时接受一个gcn_specs对象，调用build方法即可构建相应的图卷及网络

## 使用示例

```python
# specify GCN parameters
specs = gcn_specs()
specs.n_gconv_layers = 3
specs.laplacians = [np.ones([49, 49], dtype='int32')] * 3
specs.n_gconv_filters = [32, 64, 128]
specs.polynomial_orders = [4, 4, 4]
specs.pooling_sizes = [1, 1, 1]
specs.fc_dims = [256, 128]
specs.bias_type = 'per_filter'
specs.pool_fn = tf.nn.max_pool
specs.activation_fn = None
specs.regularize = False

# build GCN
gcn = GCN(specs, is_training=self.is_training)
gcn_out, regularizers = gcn.build(self.features, self.ph_dropout)
```

