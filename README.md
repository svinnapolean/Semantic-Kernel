# Semantic-Kernel
sample Semantic Kernel setup showing how to create a basic AI agent that integrates a skill, a tool, and memory, orchestrates them, and uses an LLM (like Azure OpenAI GPT). This will be in C#, which aligns well with your .NET experience.


The `Program.cs` file in your [`Semantic-Kernel`](https://github.com/svinnapolean/Semantic-Kernel/blob/main/semantic/Program.cs) repo defines a modular agentic AI workflow using Semantic Kernel and Azure OpenAI. Here's a breakdown you can use for your README or markdown documentation:

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
