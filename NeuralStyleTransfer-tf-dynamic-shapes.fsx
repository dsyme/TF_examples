#r "netstandard"
#r "lib/Argu.dll"
#r "lib/TensorFlowSharp.dll"
#load "shared/NNImpl.fsx"
#load "shared/NNOps.fsx"
#load "shared/NPYReaderWriter.fsx"
#load "shared/ImageWriter.fsx"

#nowarn "49"

open Argu
open NPYReaderWriter
open System
open System.IO
open TensorFlow
open TensorFlow

if not System.Environment.Is64BitProcess then System.Environment.Exit(-1)

type Argument =
    | [<AltCommandLine([|"-s"|])>] Style of string
    interface IArgParserTemplate with
        member this.Usage =
            match this with
            |  Style _ -> "Specify a style of painting to use."

let style = ArgumentParser<Argument>().Parse(fsi.CommandLineArgs.[1..]).GetResult(<@ Argument.Style @>, defaultValue = "rain")

fsi.AddPrinter(fun (x:TFGraph) -> sprintf "TFGraph %i" (int64 x.Handle))

let pretrained_dir = Path.Combine(__SOURCE_DIRECTORY__,"pretrained")

let example_dir = Path.Combine(__SOURCE_DIRECTORY__,"examples")

type Shape(shape: int[]) =

    member __.Item with get v = shape.[v]

    member __.Dimensions = shape

    override __.ToString() = sprintf "%A" shape

    member shape.AsTFShape() = TFShape(shape.Dimensions |> Array.map int64)

    member shape.AsTFTensor() = shape.AsTFShape().AsTensor()

    static member NewInferred() = failwith "tbd"

let equivShapes (shape1: Shape) (shape2: Shape) = 
    if shape1.Dimensions = shape2.Dimensions then 
        shape1 
    else 
        failwithf "mismatched shapes: %A and %A" shape1 shape2 

type V<'T>(shape: Shape, eval: (TFGraph -> TFOutput)) =

    member __.Apply(graph) = eval graph

    static member (+) (v1: V<'T>, v2: V<'T>) : V<'T> = V (equivShapes v1.Shape v2.Shape, fun graph -> graph.Add(v1.Apply(graph), v2.Apply(graph)))

    static member (-) (v1: V<'T>, v2: V<'T>) : V<'T> = V (equivShapes v1.Shape v2.Shape, fun graph -> graph.Sub(v1.Apply(graph), v2.Apply(graph)))

    static member ( * ) (v1: V<'T>, v2: V<'T>) : V<'T> = (V (equivShapes v1.Shape v2.Shape, fun graph -> graph.Mul(v1.Apply(graph), v2.Apply(graph))))

    static member (/) (v1: V<'T>, v2: V<'T>) : V<'T> = (V (equivShapes v1.Shape v2.Shape, fun graph -> graph.Div(v1.Apply(graph), v2.Apply(graph))))

    static member Sqrt (v: V<'T>) : V<'T> = (V (v.Shape, fun graph -> graph.Sqrt(v.Apply(graph))))

    static member Tanh (v: V<'T>) : V<'T> = (V (v.Shape, fun graph -> graph.Tanh(v.Apply(graph))))

    static member Tan (v: V<'T>) : V<'T> =  (V (v.Shape, fun graph -> graph.Tan(v.Apply(graph))))

    member __.Shape : Shape = shape

type TF() =

    static member Const (shape: Shape, value: 'T) : V<'T> = 
        V (shape, fun graph -> graph.Reshape(graph.Const(new TFTensor(value)), graph.Const(shape.AsTFTensor())))

    static member ConstArray (shape: Shape, value: 'T[]) : V<'T> = 
        V (shape, fun graph -> graph.Reshape(graph.Const(new TFTensor(value)), graph.Const(shape.AsTFTensor())))

    static member TruncatedNormal (shape: Shape) : V<double> = 
        V (shape, fun graph -> graph.TruncatedNormal(graph.Const(shape.AsTFTensor()), TFDataType.Double ))

    static member Variable (value: V<'T>) : V<'T> = 
        V (value.Shape, fun graph -> graph.Variable(value.Apply(graph)).Read)

    static member Conv2D (input: V<'T>, filters: V<'T>, ?stride: int, ?padding: string) : V<'T> = 
    //[N,H,W,C], filters: V[C;COut;F]) -> V[N,H,W,COut] 
        let stride = defaultArg stride 1
        let padding = defaultArg padding "SAME"
        let inputShape = input.Shape
        let filtersShape = filters.Shape
        let outputShape = Shape [| inputShape.[0]; inputShape.[1]; inputShape.[2]; filtersShape.[1] |]
        V (outputShape, fun graph -> graph.Conv2D(input.Apply(graph), filters.Apply(graph),strides = [|1L;int64 stride;int64 stride;1L|], padding=padding))

    static member Conv2DBackpropInput(filters: V<'T>, out_backprop: V<'T>, ?stride: int, ?padding: string) : V<'T> = 
        let stride = defaultArg stride 1
        let padding = defaultArg padding "SAME"
        let shape = out_backprop.Shape
        let filtersShape = filters.Shape
        let outputShape = Shape [| shape.[0]; shape.[1]; shape.[2]; filtersShape.[1] |]
        V (outputShape, fun graph -> 
           let batch_size = shape.[0]
           let rows = shape.[1]
           let cols = shape.[2]
           let num_filters = filtersShape.[2]
           let output_shape = Shape [|batch_size; rows*stride; cols*stride; num_filters |]
           let input_sizes = graph.Const(output_shape.AsTFTensor())
           graph.Conv2DBackpropInput(input_sizes, filters.Apply(graph), out_backprop.Apply(graph), strides = [|1L;int64 stride;int64 stride;1L|], padding=padding))
         // , output_shape=[input.Shape.[0],H*Stride,W*Stride,filters.Shape.[3]]

    static member ClipByValue (input: V<'T>, low: V<'T>, high: V<'T>) : V<'T> = 
        let outputShape = equivShapes (equivShapes input.Shape low.Shape) high.Shape
        V (outputShape, fun graph -> graph.ClipByValue(input.Apply(graph), low.Apply(graph), high.Apply(graph)))

    static member Moments(shape: Shape, input: V<'T>) : V<'T> * V<'T> = 
        let outputShape = shape
        // TODO: we make two Moments nodes here
        // TODO: keep_dims
        V (outputShape, fun graph -> fst (graph.Moments(input.Apply(graph), graph.Const(shape.AsTFTensor())))),
        V (outputShape, fun graph -> snd (graph.Moments(input.Apply(graph), graph.Const(shape.AsTFTensor()))))

    static member Relu(input: V<'T>) : V<'T> = 
        let outputShape = input.Shape
        V (outputShape, fun graph -> graph.Relu(input.Apply(graph)))

    static member DecodeJpeg(contents:V<string>, ?channels: int) : V<int> = // V[int,H,W,C]
        let channels = defaultArg channels 3 // CHECK ME
        let outputShape = Shape [| -1; -1; channels |]
        V (outputShape, fun graph -> graph.DecodeJpeg(contents=contents.Apply(graph), channels=Nullable(3L)))

    static member Cast<'T, 'T2>(input: V<'T>, dt) : V<'T2> = 
        let outputShape = input.Shape
        V (outputShape, fun graph -> graph.Cast(input.Apply(graph), dt))

    static member CreateString(value: byte[]) : V<string> = 
        let outputShape = Shape [| 1 |]
        V (outputShape, fun graph -> graph.Const(TFTensor.CreateString(value)))

    static member ExpandDims(value: V<'T>, [<ParamArray>] dims: int[]) : V<'T> = 
        let outputShape = Shape dims
        V (outputShape, fun graph -> graph.ExpandDims(value.Apply(graph), graph.Const(outputShape.AsTFTensor())))
    //TF.ExpandDims[Dim](value: V[shape]) : V[Expanded(Dim,shape)]

    static member Run(value: V<'T>) : TFTensor = 
        let sess = new TFSession()
        let graph = sess.Graph
        let node = value.Apply(graph)
        sess.Run([||],[||],[|node|]).[0]

type TensorFlow = ReflectedDefinitionAttribute

type TFBuilder() =
    member x.Return(v: 'T) = v
    //member x.Zero() = V()
    //member x.Run(v) = v

//type NumericLiteralG() =
//    member x.FromString(s: string) : V<double> = failwith "tbd"
    
let tf = TFBuilder()

let shape (ints: int list) = Shape(Array.ofSeq ints)

let v (d:double) : V<double> = 
    let shape = Shape.NewInferred ()
    V (shape, (fun graph -> graph.Const(new TFTensor(d))))

[<TensorFlow>]
module NeuralStyles = 

    let conv_init_vars (in_channels: int, out_channels:int, filter_size:int) =
        tf { let truncatedNormal = TF.TruncatedNormal(shape [filter_size; filter_size; in_channels; out_channels])
             return TF.Variable (truncatedNormal * v 0.1) }

    let instance_norm (input: V<double>) =
        tf { let C = input.Shape.[3]
             let mu, sigma_sq = TF.Moments (shape [1;2], input)
             let shift = TF.Variable (v 0.0)
             let scale = TF.Variable (v 1.0) 
             let epsilon = v 0.001
             let normalized = (input - mu) / sqrt (sigma_sq + epsilon)
             return scale * normalized + shift }

    let conv_layer (input: V<double>, COut, F, stride, is_relu) = 
        tf { let C = input.Shape.[3]
             let filters = conv_init_vars (C, COut, F)
             let x = TF.Conv2D (input, filters, stride=stride)
             let x = instance_norm x
             if is_relu then 
                 return TF.Relu x 
             else 
                 return x }
    let residual_block (input, F) = 
        tf { let tmp = conv_layer(input, 128, F, 1, true)
             return input + conv_layer(tmp, 128, F, 1, false) }

    let conv2D_transpose (input, filter, stride) = 
        tf { return TF.Conv2DBackpropInput(filter, input, stride, padding = "SAME") }
  
    let conv_transpose_layer (input: V<double>, COut, F, Stride) =
        tf { let C = input.Shape.[3]
             let filters = conv_init_vars (C, COut, F)
             return TF.Relu (instance_norm (conv2D_transpose (input, filters, Stride)))
           }

    let to_pixel_value (input: V<double>) = 
        tf { return tanh input * v 150.0 + (v 255.0 / v 2.0) }

    // The style-transfer tf
    let PretrainedFFStyleVGG input_img = 
        tf { let x = conv_layer (input_img, 32, 9, 1, true)
             let x = conv_layer (x, 64, 3, 2, true)
             let x = conv_layer (x, 128, 3, 2, true)
             let x = residual_block (x, 3)
             let x = residual_block (x, 3)
             let x = residual_block (x, 3)
             let x = residual_block (x, 3)
             let x = conv_transpose_layer (x, 64, 3, 2)
             let x = conv_transpose_layer (x, 32, 3, 2)
             let x = conv_layer (x, 3, 9, 1, false)
             let x = to_pixel_value x
             let x = TF.ClipByValue (x, v 0.0, v 255.0)
             return x }

    // Compute the weights path
    // let weights_path = Path.Combine(pretrained_dir, sprintf "fast_style_weights_%s.npz" style)

    // Read the weights map
    //let weights_map = 
    //    readFromNPZ((File.ReadAllBytes(weights_path)))
    //    |> Map.toArray 
    //    |> Array.map (fun (k,(metadata, arr)) -> 
    //        k.Substring(0, k.Length-4), graph.Reshape(graph.Const(new TFTensor(arr)), graph.Const(Shape(metadata.shape |> Array.map int64).AsTensor()))) 
    //    |> Map.ofArray

    // The average pixel in the decoding
    let mean_pixel shape = 
        tf { return TF.ConstArray (shape, [| 123.68; 116.778; 103.939 |]) }

    // The decoding tf
    let img input_string = 
        tf { 
            let jpg = TF.DecodeJpeg(input_string)
            let decoded = TF.Cast<_, double>(jpg, TFDataType.Double)
            let preprocessed = decoded - mean_pixel decoded.Shape
            let expanded = TF.ExpandDims(preprocessed, 0)
            return expanded
        }

    // Tensor to read the input
    let img_tf = 
        tf { let bytes = File.ReadAllBytes(Path.Combine(example_dir,"chicago.jpg"))
             return TF.CreateString (bytes) } 

    // Run the decoding
    let img_tensor = img img_tf

    // Run the style transfer
    let img_styled = TF.Run (PretrainedFFStyleVGG img_tensor)

    // NOTE: Assumed NHWC dataformat
    let tensorToPNG(batchIndex:int) (imgs:TFTensor) =
        if imgs.TensorType <> TFDataType.Float then failwith "type unsupported"
        let shape = imgs.Shape |> Array.map int 
        let _N, H, W, C = shape.[0], shape.[1], shape.[2], shape.[3]
        let pixels = 
            [|
                let res_arr = imgs.GetValue() :?> Array
                for h in 0..H-1 do
                    for w in 0..W-1 do
                        let getV(c) = byte <| Math.Min(255.f, Math.Max(0.f, (res_arr.GetValue(int64 batchIndex, int64 h, int64 w, int64 c) :?> float32)))
                        yield BitConverter.ToInt32([|getV(0); getV(1); getV(2); 255uy|], 0) // NOTE: Channels are commonly in RGB format
            |]
        ImageWriter.RGBAToPNG(H,W,pixels)

    // Write the result
    File.WriteAllBytes(Path.Combine(__SOURCE_DIRECTORY__, sprintf "chicago_in_%s_style.png" style), tensorToPNG 0 img_styled)
