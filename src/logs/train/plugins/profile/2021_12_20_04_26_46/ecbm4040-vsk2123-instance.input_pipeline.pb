	[닄?uc@[닄?uc@![닄?uc@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-[닄?uc@/M???#@1??:q?^a@A?LN????I??n??@*	sh??|#d@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatz?,C???!o??Q@@)??SV????1???h9@:Preprocessing2U
Iterator::Model::ParallelMapV2?1?????!??F?+8@)?1?????1??F?+8@:Preprocessing2F
Iterator::Modeld${??!??!3
9??D@)?-?熦??1DK+?]1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicecFx{??!?Fb??%@)cFx{??1?Fb??%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatea????!ڀC?1?3@)??{?E{??14r@J?C!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;oc?#Շ?!7?9?i?@);oc?#Շ?17?9?i?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?8?????!????T;M@)?????P}?1b?9??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??+?z???!KA?ATc5@)cAJh?1\?&r??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI???dz%@Q?}m?PV@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	/M???#@/M???#@!/M???#@      ??!       "	??:q?^a@??:q?^a@!??:q?^a@*      ??!       2	?LN?????LN????!?LN????:	??n??@??n??@!??n??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???dz%@y?}m?PV@