?	;???d@;???d@!;???d@	q??~F??q??~F??!q??~F??"w
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
	6u?(@6u?(@!6u?(@      ??!       "	E(b?)b@E(b?)b@!E(b?)b@*      ??!       2	4d<J%<??4d<J%<??!4d<J%<??:	l???ڿ@l???ڿ@!l???ڿ@B      ??!       J	?¼Ǚ????¼Ǚ???!?¼Ǚ???R      ??!       Z	?¼Ǚ????¼Ǚ???!?¼Ǚ???b      ??!       JGPUYq??~F??b qX????%@y?ȟ?/V@?"<
 sequential/spectral__pool/IFFT2DIFFT2D??
?cV??!??
?cV??"i
=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter]??????!*|kɬ???0"g
<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput 4nN????!KQ8?E??0"H
-gradient_tape/sequential/spectral__pool/FFT2DFFT2DX?k?{-??!?J,.?P??"i
=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?س????!A?Ѱ??0"g
<gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInputa???N??!7ݿ??z??0":
sequential/conv2d_1/Relu_FusedConv2D??Y#7??!J_?:???"J
.gradient_tape/sequential/spectral__pool/IFFT2DIFFT2D1Xjb?o??!P?A'?/??"i
=gradient_tape/sequential/conv2d_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter(??N??!??iP??0":
sequential/conv2d_2/Relu_FusedConv2D?i?;藐?!	y?fc??Q      Y@Y"h8???@a~ylE??W@q?h1?m?:@y!?<%\?"?

both?Your program is POTENTIALLY input-bound because 7.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?26.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 