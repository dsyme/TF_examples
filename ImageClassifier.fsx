#r "netstandard"
#r "lib/HDF.PInvoke.dll"
#r "lib/TensorFlowSharp.dll"
#load "shared/HDF5ReaderWriter.fsx"
#nowarn "760"
open TensorFlow
open System
open System.IO
open System.Collections.Generic

if not System.Environment.Is64BitProcess then System.Environment.Exit(-1)

let pretrained_dir = Path.Combine(__SOURCE_DIRECTORY__,"..","..","pretrained")
let paket_files_dir = Path.Combine(__SOURCE_DIRECTORY__,"..","..","paket-files")
let weights_path = Path.Combine(paket_files_dir, "github.com","resnet50_weights_tf_dim_ordering_tf_kernels.h5")
let labels_path = Path.Combine(pretrained_dir,"imagenet1000.txt")
let example_dir = Path.Combine(__SOURCE_DIRECTORY__, "..","..","examples")
let label_map = File.ReadAllLines(labels_path)


let sess = new TFSession()
// NOTE: Graph.ToString() returns the whole protobuf as txt to console
// NOTE: fsi does not type check in Ionide. This can be ignored.
fsi.AddPrinter(fun (x:TFGraph) -> sprintf "TFGraph %i" (int64 x.Handle))

let graph = sess.Graph



let buildResnet(graph:TFGraph,weights_path:string) =
    // NOTE: This behaviour should be built into graph
    // NOTE: This is needed as by default the graph will use the same block for each invocation
    let withScope =
        let nameCount = Dictionary<string,int>()
        fun (name:string) -> 
            let namePrime = 
                if nameCount.ContainsKey(name) then 
                    nameCount.[name] <- nameCount.[name] + 1
                    sprintf "%s_%i" name nameCount.[name] 
                else nameCount.Add((name,1)); name
            graph.WithScope(namePrime)


    use h5 = HDF5.OpenRead(weights_path) 
    let getWeights(name:string) =
        let data, shape = h5.Read<float32>(name)
        graph.Reshape(graph.Const(new TFTensor(data)), graph.Const(TFShape(shape |> Array.ofList).AsTensor()))

    let get_conv_tensor(conv_name:string) = getWeights(sprintf "%s/%s_W:0" conv_name conv_name)

    let make_batch_norm(bn_name:string) bnx = 
        use ns = withScope("batchnorm")
        let getT(nm) = getWeights(sprintf "%s/%s_%s:0" bn_name bn_name nm)
        let moving_variance = getT("running_std")
        let gamma           = getT("gamma") // AKA scale
        let moving_mean     = getT("running_mean")
        let beta            = getT("beta")
        let (fbn,_,_,_,_) = graph.FusedBatchNorm(bnx,gamma,beta,mean=moving_mean,
                             variance=moving_variance, epsilon=Nullable(0.00001f),
                             is_training=Nullable(false), data_format="NHWC").ToTuple()
        fbn                         

    let res_block(stage:int, 
                  block:char, 
                  is_strided:bool, 
                  conv_shortcut:bool)
                  input_tensor:TFOutput =
        use scope = withScope("resblock")
        let conv_name_base = sprintf "res%i%c_branch" stage block
        let bn_name_base = sprintf "bn%i%c_branch" stage block
        let conv(postfix,is_strided:bool) cx =
            use ns = withScope("conv")
            let conv_name = sprintf "res%i%c_branch" stage block
            let strides = if is_strided then [|1L;2L;2L;1L|] else [|1L;1L;1L;1L|]
            graph.Conv2D(cx,
                         get_conv_tensor(conv_name_base + postfix),
                         strides,
                         padding="SAME",
                         data_format="NHWC",
                         dilations=[|1L;1L;1L;1L|],
                         operName=conv_name + postfix)
        let right = 
            input_tensor
            |> conv("2a",is_strided)
            |> make_batch_norm(bn_name_base + "2a")
            |> graph.Relu
            |> conv("2b",false) // This is the only 3x3 conv
            |> make_batch_norm(bn_name_base + "2b")
            |> graph.Relu
            |> conv("2c",false)
            |> make_batch_norm(bn_name_base + "2c")
        let left = 
            if conv_shortcut then 
                input_tensor |> conv("1",is_strided) |> make_batch_norm(bn_name_base + "1")
            else input_tensor
        graph.Add(right,left) |> graph.Relu
        
    let input_placeholder = 
        graph.Placeholder(TFDataType.Float, 
                          shape=TFShape(-1L,-1L,-1L,3L), 
                          operName="new_input")

    /// TODO make this simpler with helper functions
    let paddings = graph.Reshape(graph.Const(new TFTensor([|0;0;3;3;3;3;0;0|])), graph.Const(TFShape(4L,2L).AsTensor()))
    let padded_input = graph.Pad(input_placeholder,paddings, "CONSTANT")
    let build_stage(stage:int,blocks:string) (x:TFOutput) =
        blocks.ToCharArray() 
        |> Array.fold (fun x c -> res_block(stage,c,c='a' && stage<>2,c='a')(x)) x
    let toAxis (xs:int list) : Nullable<TFOutput> = 
        Nullable(graph.Const(new TFTensor(xs |> Array.ofList),TFDataType.Int32))
    let softmax = 
        graph.Conv2D(padded_input, 
                     get_conv_tensor("conv1"),
                     [|1L;2L;2L;1L|],
                     padding="VALID",
                     data_format="NHWC",
                     operName="conv1")
        |> make_batch_norm("bn_conv1") 
        |> graph.Relu
        |> fun x -> graph.MaxPool(x,[|1L;3L;3L;1L|],[|1L;2L;2L;1L|],padding="SAME",data_format="NHWC")
        |> build_stage(2,"abc")
        |> build_stage(3,"abcd")
        |> build_stage(4,"abcdef")
        |> build_stage(5,"abc")
        |> fun x -> graph.ReduceMean(x,axis=([1;2] |> toAxis)) 
        |> fun x -> graph.MatMul(x,getWeights("fc1000/fc1000_W:0"))
        |> fun x -> graph.Add(x, getWeights("fc1000/fc1000_b:0"))
        |> fun x -> graph.Softmax(x)
    (input_placeholder,softmax)


/// This is from TensorflowSharp (Examples/ExampleCommon/ImageUtil.cs)
/// It's intended for inception but used here for resnet as an example
/// of this type of functionality 
let construtGraphToNormalizeImage(destinationDataType:TFDataType) =
    let W = 224
    let H = 224
    let Mean = 117.0f
    let Scale = 1.0f
    let input = graph.Placeholder(TFDataType.String)
    let loaded_img = graph.Cast(graph.DecodeJpeg(contents=input,channels=Nullable(3L)),TFDataType.Float)
    let expanded_img = graph.ExpandDims(input=loaded_img, dim = graph.Const(TFTensor(0)))
    let resized_img = graph.ResizeBilinear(expanded_img,graph.Const(TFTensor([|W;H|])))
    let final_img = graph.Div(graph.Sub(resized_img, graph.Const(TFTensor([|Mean|]))), graph.Const(TFTensor([|Scale|])))
    (input,graph.Cast(final_img,destinationDataType))

let img_input,img_output = construtGraphToNormalizeImage(TFDataType.Float)

let (input,output) = buildResnet(graph,weights_path)

let classifyFile(path:string) =
    let createTensorFromImageFile(file:string,destinationDataType:TFDataType) =
        let tensor = TFTensor.CreateString(File.ReadAllBytes(file))
        sess.Run(runOptions = null, inputs = [|img_input|], inputValues = [|tensor|], outputs = [|img_output|]).[0]
    let example = createTensorFromImageFile(path, TFDataType.Float)
    let index = graph.ArgMax(output,graph.Const(TFTensor(1)))
    let res = sess.Run(runOptions = null, inputs = [|input|], inputValues = [|example|], outputs = [|index|])
    label_map.[res.[0].GetValue() :?> int64[] |> Array.item 0 |> int]

classifyFile(Path.Combine(example_dir,"example_0.jpeg"))
classifyFile(Path.Combine(example_dir,"example_1.jpeg"))
classifyFile(Path.Combine(example_dir,"example_2.jpeg"))