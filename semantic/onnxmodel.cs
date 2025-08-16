using System;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

public class onnxModel
{
    public string load_and_invoke_onnx ()
    {
        // Load the ONNX model
        using var session = new InferenceSession(@"D:\source\onnx1\linear_model.onnx");

        // Prepare input (1D example: x = 7)
        var inputData = new float[] { 7f };
        var inputTensor = new DenseTensor<float>(inputData, new int[] { 1, 1 });

        // Create input container
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("float_input", inputTensor)
        };

        // Run inference
        using var results = session.Run(inputs);
        var prediction = results.First().AsEnumerable<float>().First();

        //Console.WriteLine($"Prediction for x=7: {prediction}");
        return prediction.ToString();
    }

}
