	I?<?+?U@I?<?+?U@!I?<?+?U@	NaV???NaV???!NaV???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6I?<?+?U@??Rb%@1??I??Q@A?fI-???I??:??D@Y????:s??*	+??_@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?L2r???!???u@@)3??bb???1l????i;@:Preprocessing2U
Iterator::Model::ParallelMapV22Xq??0??!맄??4@)2Xq??0??1맄??4@:Preprocessing2F
Iterator::Model??!?Q*??!?n{(@7C@)????#??1W5r?j?1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??MbX??!$?W.Z#@)??MbX??1$?W.Z#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatez5@i?Q??!܈?ϑ2@)?2p@K??1??$q?!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipĵ??^(??!_??׿?N@)?#c????1????j @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?*5{?x?!C
??c@)?*5{?x?1C
??c@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap*?TPQ???!?(?6Y5@)|??8Gm?1?e?3;@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 12.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?7.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9NaV???I`??<?T3@Q?y??T@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??Rb%@??Rb%@!??Rb%@      ??!       "	??I??Q@??I??Q@!??I??Q@*      ??!       2	?fI-????fI-???!?fI-???:	??:??D@??:??D@!??:??D@B      ??!       J	????:s??????:s??!????:s??R      ??!       Z	????:s??????:s??!????:s??b      ??!       JGPUYNaV???b q`??<?T3@y?y??T@