	;???d@;???d@!;???d@	q??~F??q??~F??!q??~F??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6;???d@6u?(@1E(b?)b@A4d<J%<??Il???ڿ@Y?¼Ǚ???*	?G?zL?@2F
Iterator::Modell??F????!?~?0?5W@)??ʡ???1?E??LV@:Preprocessing2U
Iterator::Model::ParallelMapV2?5??Wt??!?{2?$@)?5??Wt??1?{2?$@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?p?GRң?!LW?)??@)T?????1݁?~??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?O:?`???!?eX?c9@).</???1dh???/??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?bE?a??!mc.?)C??)?bE?a??1mc.?)C??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?m????!ߪ1V????)?m????1ߪ1V????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip9??!??!?????@)M?~2Ƈy?1?px??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?L?^?i??!	Ep???@)???5??r?1??8?s??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 7.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9q??~F??IX????%@Q?ȟ?/V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	6u?(@6u?(@!6u?(@      ??!       "	E(b?)b@E(b?)b@!E(b?)b@*      ??!       2	4d<J%<??4d<J%<??!4d<J%<??:	l???ڿ@l???ڿ@!l???ڿ@B      ??!       J	?¼Ǚ????¼Ǚ???!?¼Ǚ???R      ??!       Z	?¼Ǚ????¼Ǚ???!?¼Ǚ???b      ??!       JGPUYq??~F??b qX????%@y?ȟ?/V@