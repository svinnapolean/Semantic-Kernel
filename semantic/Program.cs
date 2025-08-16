using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.AzureOpenAI;
using Microsoft.SemanticKernel.Memory;
using Microsoft.SemanticKernel.Connectors.InMemory;
using System;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.VectorData;
using OpenAI.VectorStores;

class Program
{
    static async Task Main(string[] args)
    {
        // 1. Configure kernel with Azure OpenAI
        var builder = Kernel.CreateBuilder();
        builder.AddAzureOpenAIChatCompletion(
            deploymentName: "gpt-4",
            endpoint: "<YOUR_ENDPOINT>",
            apiKey: "<YOUR_AZURE_OPENAI_KEY>"
        );

        // Add the In-Memory Vector Store
        builder.Services.AddInMemoryVectorStore();

        var kernel = builder.Build();

        // 2. Define inline functions (skills)
        var regressionFunction = KernelFunctionFactory.CreateFromMethod(async (KernelArguments args) =>
        {
            var input = args["features"]?.ToString() ?? "";
           
            return (input.Length * 0.5).ToString();
        }, "RegressionSkill");

        var decisionTreeFunction = KernelFunctionFactory.CreateFromMethod(async (KernelArguments args) =>
        {
            var input = args["features"]?.ToString() ?? "";
            return input.Length > 5 ? "High" : "Low";
        }, "DecisionTreeSkill");

        var clusteringFunction = KernelFunctionFactory.CreateFromMethod(async (KernelArguments args) =>
        {
            var input = args["features"]?.ToString() ?? "";
            return input.Length % 2 == 0 ? "ClusterA" : "ClusterB";
        }, "ClusteringSkill");

        var onnxlinearregressionFunction = KernelFunctionFactory.CreateFromMethod(async (KernelArguments args) =>
        {
             onnxModel model = new onnxModel();
            string results_onnx = model.load_and_invoke_onnx();
            return results_onnx;
        }, "ONNXlinearregressionSkill");

        kernel.Plugins.AddFromFunctions("MLTools", [regressionFunction, decisionTreeFunction, clusteringFunction, onnxlinearregressionFunction]);

        // 3. Agentic workflow
        var args_in = new KernelArguments { ["features"] = "user_input_data_example" };

        var regressionResult = await kernel.InvokeAsync("MLTools", "RegressionSkill", args_in);
        var decisionResult = await kernel.InvokeAsync("MLTools", "DecisionTreeSkill", args_in);
        var clusterResult = await kernel.InvokeAsync("MLTools", "ClusteringSkill", args_in);
        var onnxResult = await kernel.InvokeAsync("MLTools", "ONNXlinearregressionSkill", args_in);

        var combinedResult = $"\tRegression: {regressionResult.GetValue<string>()}, " +
                             $"\tDecision: {decisionResult.GetValue<string>()}, " +
                             $"\tCluster: {clusterResult.GetValue<string>()}"+
                             $"\tONNXLinearRegression: {onnxResult.GetValue<string>()}"; 

        // //  Use the IVectorStore from DI
        // var vectorStore = kernel.Services.GetRequiredService<VectorStore>();
        // var memory = new MemoryBuilder().WithVectorStore(vectorStore).Build();

        // await memory.SaveInformationAsync("LastAnalysis", combinedResult, "MLResults");

        Console.WriteLine($"Final Agentic AI Result: {combinedResult}");
    }

}