# Semantic-Kernel with ONNX
sample Semantic Kernel setup showing how to create a basic AI agent that integrates a skill, a tool, and memory, orchestrates them, and uses an LLM (like Azure OpenAI GPT). This will be in C#, which aligns well with your .NET experience.


The `Program.cs` file in [`Semantic-Kernel`](https://github.com/svinnapolean/Semantic-Kernel/blob/main/semantic/Program.cs) repo defines a modular agentic AI workflow using Semantic Kernel and Azure OpenAI.

---

## 🧠 Program Overview

This C# program demonstrates how to build an agentic AI pipeline using Semantic Kernel with inline ML-style functions and ONNX model integration.

---

## ⚙️ Key Components

### 1. **Kernel Configuration**
```csharp
builder.AddAzureOpenAIChatCompletion(...);
builder.Services.AddInMemoryVectorStore();
```
- Connects to Azure OpenAI (`gpt-4`).
- Adds an in-memory vector store for semantic memory.

---

### 2. **Inline ML Functions (Skills)**
Defined using `KernelFunctionFactory.CreateFromMethod`:
- **RegressionSkill**: Returns a scaled value based on input length.
- **DecisionTreeSkill**: Classifies input as "High" or "Low".
- **ClusteringSkill**: Assigns input to `ClusterA` or `ClusterB`.
- **ONNXlinearregressionSkill**: Loads and invokes an ONNX model via `onnxModel.load_and_invoke_onnx()`.

These are grouped under the plugin name `"MLTools"`.

---

### 3. **Agentic Workflow Execution**
```csharp
var args_in = new KernelArguments { ["features"] = "user_input_data_example" };
var regressionResult = await kernel.InvokeAsync("MLTools", "RegressionSkill", args_in);
...
```
- Invokes each skill with the same input.
- Aggregates results into a final output string.

---

### 4. **Optional Semantic Memory**
Commented out but ready for use:
```csharp
var memory = new MemoryBuilder().WithVectorStore(vectorStore).Build();
await memory.SaveInformationAsync("LastAnalysis", combinedResult, "MLResults");
```
- Stores results in vector memory using DI-injected `IVectorStore`.

---

## 📤 Output
```plaintext
Final Agentic AI Result: 
    Regression: 12.5, 
    Decision: High, 
    Cluster: ClusterA
    ONNXLinearRegression: <model_output>
```

---


## 🧠 How to Load and Run an ONNX Model in .NET

### ✅ Prerequisites
- Install the NuGet package: `Microsoft.ML.OnnxRuntime`
```bash
dotnet add package Microsoft.ML.OnnxRuntime
```

---

### 📦 Code Breakdown

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

public class onnxModel {
    public string load_and_invoke_onnx () {
        // 1. Load the ONNX model
        using var session = new InferenceSession(@"D:\source\onnx1\linear_model.onnx");

        // 2. Prepare input tensor (example: x = 7)
        var inputData = new float[] { 7f };
        var inputTensor = new DenseTensor<float>(inputData, new int[] { 1, 1 });

        // 3. Create input container
        var inputs = new List<NamedOnnxValue> {
            NamedOnnxValue.CreateFromTensor("float_input", inputTensor)
        };

        // 4. Run inference
        using var results = session.Run(inputs);
        var prediction = results.First().AsEnumerable<float>().First();

        return prediction.ToString();
    }
}
```

---

### 🧪 What It Does
- Loads a local ONNX model (`linear_model.onnx`)
- Creates a single float input tensor (`x = 7`)
- Feeds it into the model using `NamedOnnxValue`
- Runs inference and extracts the first prediction

---

### 📍 Notes
- `"float_input"` must match the input name defined in the ONNX model.
- The model path is hardcoded—consider making it configurable.
- You can extend this to batch inputs or multi-dimensional tensors.

---
