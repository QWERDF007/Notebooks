# [NVIDIA Deep Learning TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)

## 摘要

此 NVIDIA TensorRT 开发者指南演示了如何使用 C++ 和 Python APIs 实现最常见的深度学习层。它展示了如何使用提供的解析器获取使用深度学习框架构建的现有模型，并构建 TensorRT 引擎。开发人员指南还提供了逐步说明常见用户任务的说明，例如创建 TensorRT 网络定义，调用 TensorRT 构建器，序列化和反序列化以及如何使用 C++ 或 Python API 提供数据并执行推理。

## 1. 介绍





## 2. TensorRT 的功能



## 6. Advanced Topics

### 6.1. Version Compatibility





## 9. 使用自定义层扩展 TensorRT

NVIDIA TensorRT 支持许多类型的层，其功能不断扩展；但是，在某些情况下，支持的层可能无法满足模型的特定需求。在这种情况下，可以通过实现自定义层 (通常称为插件) 来扩展 TensorRT。

TensorRT 包含可以加载到应用程序中的插件。有关开源插件列表，请参见 [GitHub: TensorRT](https://github.com/NVIDIA/TensorRT/tree/main/plugin#tensorrt-plugins) 插件。

要在应用程序中使用 TensorRT 插件，必须加载 `libnvinfer_plugin.so` (Windows 上为 `nvinfer_plugin.dll` ) 库，并通过在应用程序代码中调用 `initLibNvInferPlugins` 来注册所有插件。有关这些插件的更多信息，请参见 [NvInferPlugin.h](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/_nv_infer_plugin_8h.html) 文件以供参考。

如果这些插件不符合您的需求，则可以编写并添加自己的插件。

### 9.1. 使用 C++ API 添加自定义层

你可以通过从 TensorRT 的插件基类之一派生自定义层。

从插件的基类中派生您的插件类。它们在支持具有不同类型/格式或动态形状的I/O方面具有不同的表现力。下表总结了基类，按表达能力从最低到最高排序。

**注意：** 如果插件用于一般用途，请提供一个 FP32 实现，以使其能够正确地与任何网络一起运行。

|                                                              | 引入 TensorRT 中的版本？       | 混合 I/O 格式/类型 | 动态形状？ | 支持隐式/显式批处理模式？ |
| ------------------------------------------------------------ | ------------------------------ | ------------------ | ---------- | ------------------------- |
| [IPluginV2Ext](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_i_plugin_v2_ext.html) | 5.1 (自 TensorRT 8.5 起已弃用) | 有限的             | No         | 隐式和显式批处理模式      |
| [IPluginV2IOExt](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_i_plugin_v2_i_o_ext.html) | 6.0.1                          | 一般的             | No         | 隐式和显式批处理模式      |
| [IPluginV2DynamicExt](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_i_plugin_v2_dynamic_ext.html) | 6.0.1                          | 一般的             | Yes        | 仅显式批处理模式          |

为了在网络中使用插件，您必须首先使用 TensorRT 的 `PluginRegistry` (C ++, Python) 注册它。您为插件注册工厂类的一个实例，该工厂类派生自 `PluginCreator` (C ++, Python) ，而不是直接注册插件。插件创建器类还提供有关插件的其他信息：其名称、版本和插件字段参数。

有两种方法可以使用注册表注册插件： 

- TensorRT 提供了一个名为 `REGISTER_TENSORRT_PLUGIN` 的宏，用于静态注册插件创建器。请注意，`REGISTER_TENSORRT_PLUGIN` 始终在默认命名空间 ("") 下注册创建器。 
- 通过创建你自己的类似于 `initLibNvInferPlugins` 的入口动态注册，并在插件注册表上调用 `registerCreator`。这比静态注册更优，因为它可能提供更低的内存占用，并且允许在唯一命名空间下注册插件。这确保了在不同的插件库之间进行构建时没有命名冲突。

调用 `IPluginCreator :: createPlugin()` 返回类型为 `IPluginV2` 的插件对象。您可以使用 `addPluginV2 ()` 将插件添加到 TensorR T网络中，该函数使用给定的插件创建网络层。

例如，您可以按以下方式将插件层添加到网络中：

```c++
// 查找注册表中的插件
auto creator = getPluginRegistry()->getPluginCreator(pluginName, pluginVersion);
const PluginFieldCollection* pluginFC = creator->getFieldNames();
// 为插件层填充字段参数
// PluginFieldCollection *pluginData = parseAndFillFields(pluginFC, layerFields); 
// 使用layerName和插件元数据创建插件对象
IPluginV2 *pluginObj = creator->createPlugin(layerName, pluginData);
// 插件添加到TensorRT网络中
auto layer = network.addPluginV2(&inputs[0], int(inputs.size()), pluginObj);
… (build rest of the network and serialize engine)
// 销毁插件对象
pluginObj->destroy()
… (free allocated pluginData)
```

**注意：** 前面描述的 `createPlugin` 方法在堆上创建一个新的插件对象并返回指向它的指针。确保像之前展示的那样销毁 `pluginObj`，以避免内存泄漏。 

在序列化期间，TensorRT 引擎在内部存储所有 `IPluginV2` 类型插件的插件类型、插件版本和命名空间（如果存在）。在反序列化期间，TensorRT 从插件注册表中查找插件创建器，并调用 `IPluginCreator :: deserializePlugin()`。当引擎被删除时，由引擎创建的插件对象的克隆副本在引擎调用 `IPluginV2 :: destroy()` 方法时被销毁。你有责任确保你创建的插件对象在添加到网络后被释放。 

**注意： **

- 不要序列化所有插件参数：仅序列化运行时插件所需的参数。可以省略构建时间参数。
- 按相同顺序序列化和反序列化插件参数。在反序列化期间，请验证插件参数是否已初始化为默认值或反序列化值。未初始化的参数会导致未定义的行为。 
- 如果您是汽车安全用户，则必须调用 `getSafePluginRegistry()` 而不是 `getPluginRegistry()`。您还必须使用 `REGISTER_SAFE_TENSORRT_PLUGIN` 宏而不是 `REGISTER_TENSORRT_PLUGIN`。

#### 9.1.1. 示例：使用 C++ 添加具有动态形状支持的自定义层

为了支持动态形状，您的插件必须派生自 `IPluginV2DynamicExt`。

`BarPlugin` 是一个具有两个输入和两个输出的插件，其中： 

- 第一个输出是第二个输入的副本。 
- 第二个输出是两个输入的沿第一个维度的连接，并且所有类型/格式必须相同并且是线性格式。

`BarPlugin` 必须派生如下：

```c++
class BarPlugin : public IPluginV2DynamicExt
{
	...override virtual methods inherited from IPluginV2DynamicExt.
};
```

受动态形状影响的四种方法是：

- `getOutputDimensions`
- `supportsFormatCombination`
- `configurePlugin`
- `enqueue`

`getOutputDimensions` 的重写以输入维度的符号表达式返回输出维度。您可以使用传递到 `getOutputDimensions` 的 `IExprBuilder` 从输入的表达式构建表达式。在本例中，对于索引 1，不必为第二个输出构建新表达式，因为第二个输出的维度与第一个输入的维度相同。

```c++
DimsExprs BarPlugin::getOutputDimensions(int outputIndex, 
    const DimsExprs* inputs, int nbInputs, 
    IExprBuilder& exprBuilder)
{
    switch (outputIndex)
    {
    case 0: 
    {
        // First dimension of output is sum of input first dimensions.
        DimsExprs output(inputs[0]);
        output.d[0] = exprBuilder.operation(
        	DimensionOperation::kSUM, inputs[0].d[0], inputs[1].d[0]);
	   return output;
    }
    case 1:
        return inputs[0];
    default:
         throw std::invalid_argument(“invalid output”);
}
```

`supportsFormatCombination` 的重写必须指示是否允许格式组合。接口统一将输入/输出索引为“连接”，从第一个输入开始为 0，然后按顺序排列其余的输入，然后编号输出。在本例中，输入是连接 0 和 1，输出是连接 2 和 3。

TensorRT 使用 `supportsFormatCombination` 来询问给定格式/类型组合是否适用于连接，给定较低索引连接的格式/类型。因此，重写可以假定较低索引的连接已经过审查，并关注具有索引 `pos` 的连接。

```c++
bool BarPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override
{
    assert(0 <= pos && pos < 4);
    const auto* in = inOut;
    const auto* out = inOut + nbInputs;
    switch (pos)
    {
    case 0: return in[0].format == TensorFormat::kLINEAR;
    case 1: return in[1].type == in[0].type &&
                   in[1].format == TensorFormat::kLINEAR;
    case 2: return out[0].type == in[0].type &&
                   out[0].format == TensorFormat::kLINEAR;
    case 3: return out[1].type == in[0].type &&
                   out[1].format == TensorFormat::kLINEAR;
    }
    throw std::invalid_argument(“invalid connection number”);
}
```

这里的局部变量 `in` 和 `out` 允许通过输入或输出编号而不是连接编号来检查 `inOut`。

**重要提示：** 覆盖会检查索引小于 `pos` 的连接的格式/类型，但绝不能检查索引大于 `pos` 的连接的格式/类型。该示例使用 `case 3` 检查连接 3 与连接 0，而不是使用 `case 0` 检查连接 0 与连接 3。

TensorRT 使用 `configurePlugin` 在运行时设置插件。这个插件不需要 `configurePlugin` 来做任何事情，所以它是一个 no-op：

```c++
void BarPlugin::configurePlugin(
    const DynamicPluginTensorDesc* in, int nbInputs, 
    const DynamicPluginTensorDesc* out, int nbOutputs) override
{
}
```





## 13. 性能最佳实践

### 13.1. 测量性能

在使用 TensorRT 进行任何优化工作之前，确定应该测量什么是至关重要的。没有测量，就不可能取得可靠的进展或衡量是否取得了成功。

- 延迟 (Latency) 

  网络推理的性能测量是从输入被传递到网络到输出可用所经过的时间。这是单次推理的网络延迟。较低的延迟更好。在某些应用中，低延迟是至关重要的安全要求。在其他应用中，延迟作为服务质量问题对用户直接可见。对于批量处理，延迟可能根本不重要。

- 吞吐量 (Throughput) 

  另一种性能测量是在固定时间单位内可以完成多少推理。这是网络的吞吐量。更高的吞吐量更好。更高的吞吐量表示固定计算资源的利用效率更高。对于批处理，总时间将由网络的吞吐量决定。

查看延迟和吞吐量的另一种方法是固定最大延迟并在该延迟下测量吞吐量。这样的服务质量测量可以在用户体验和系统效率之间达成合理的折衷。

在测量延迟和吞吐量之前，必须选择开始和停止计时的确切点。根据网络和应用程序，选择不同点可能是有意义的。

在许多应用中，存在一个处理管道，并且可以通过整个处理管道的延迟和吞吐量来衡量整个系统性能。由于预处理和后处理步骤非常依赖于特定应用程序，因此本节仅考虑网络推理的延迟和吞吐量。

#### 13.1.1. 墙钟时间

==墙钟时间 (计算开始和结束之间经过的时间)== 对于测量应用程序的整体吞吐量和延迟，以及将推理时间放置在更大系统的上下文中可能很有用。 C++11 在 `<chrono>` 标准库中提供了高精度计时器。例如，`std::chrono::system_clock` 表示系统范围的墙钟时间，而 `std::chrono::high_resolution_clock` 以最高可用精度测量时间。 

以下示例代码片段显示了测量网络推理主机时间：

**C++**

```c++
#include <chrono>

auto startTime = std::chrono::high_resolution_clock::now();
context->enqueueV3(stream);
cudaStreamSynchronize(stream);
auto endTime = std::chrono::high_resolution_clock::now();
float totalTime = std::chrono::duration<float, std::milli>
(endTime - startTime).count()
```

**Python**

```python
import time
from cuda import cudart
err, stream = cudart.cudaStreamCreate()
start_time = time.time()
context.execute_async_v3(stream)
cudart.cudaStreamSynchronize(stream)
total_time = time.time() - start_time
```

如果在设备上一次只有一个推理，则这可以是分析各种操作所需时间的简单方法。推理通常是异步的，因此请确保添加显式的 CUDA 流或设备同步以等待结果可用。

#### 13.1.2. CUDA Events

仅在主机上进行计时的一个问题是它需要主机/设备同步。优化过的应用程序可能在设备上并行运行许多推理，并具有重叠的数据移动。此外，同步本身会向计时测量添加一定量的噪声。 为了解决这些问题，CUDA 提供了 [Event API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html#group__CUDART__EVENT)。此 API 允许您将事件放入 CUDA 流中，GPU 在遇到它们时会标上时间戳。时间戳之间的差异可以告诉您不同操作所需的时间。

以下示例代码片段显示了计算两个 CUDA 事件之间的时间：

**C++**

```c++
cudaEvent_t start, end;
cudaEventCreate(&start);
cudaEventCreate(&end);

cudaEventRecord(start, stream);
context->enqueueV3stream);
cudaEventRecord(end, stream);

cudaEventSynchronize(end);
float totalTime;
cudaEventElapsedTime(&totalTime, start, end);
```

**Python**

```python
from cuda import cudart
err, stream = cudart.cudaStreamCreate()
err, start = cudart.cudaEventCreate()
err, end = cudart.cudaEventCreate()
cudart.cudaEventRecord(start, stream)
context.execute_async_v3(stream)
cudart.cudaEventRecord(end, stream)
cudart.cudaEventSynchronize(end)
err, total_time = cudart.cudaEventElapsedTime(start, end)
```

### 13.2. 用于性能测量的硬件/软件环境

性能测量受许多因素的影响，包括硬件环境差异 (例如机器的冷却能力) 和软件环境差异 (例如 GPU 时钟设置) 。本节总结了可能会影响性能测量的一些项目。 

请注意，涉及 nvidia-smi 的项目仅支持于 dGPU 系统，而不支持于移动系统。

#### 13.2.1. GPU 信息查询和 GPU 监控

在测量性能时，建议您同时记录和监视 GPU 状态和推理工作负载。拥有监视数据可以帮助您在看到意外的性能测量结果时识别可能的根本原因。在推理开始之前，调用 `nvidia-smi -q` 命令以获取 GPU 的详细信息，包括产品名称、功率限制、时钟设置等。然后，在推理工作负载运行时，以并行方式运行 `nvidia-smi dmon -s pcu -f <FILE> -c <COUNT>` 命令，将 GPU 时钟频率、功耗、温度和利用率打印到文件中。调用 `nvidia-smi dmon --help` 以获取有关 nvidia-smi 设备监视工具的更多选项。

#### 13.2.2. GPU 时钟锁定和浮动时钟

默认情况下，GPU 时钟频率是浮动的，这意味着当没有激活的工作负载时，时钟频率处于空闲频率，并在工作负载开始时提升到增强时钟频率。这通常是期许的行为，因为它允许 GPU 在空闲时产生较少的热量，并在有激活的工作负载时以最大速度运行。 

作为选择，您可以通过调用 `sudo nvidia-smi -lgc <freq>` 命令将时钟锁定在特定频率上 (反之亦然，您可以使用 `sudo nvidia-smi -rgc` 命令再次让时钟频率浮动) 。支持的时钟频率可以通过 `sudo nvidia-smi -q -d SUPPORTED_CLOCKS` 命令找到。锁定时钟频率后，它应该保持在该频率上，除非达到功率限制墙或温度限制墙 (过热降频) ，这将在下一节中解释。当限制开始生效时，设备的行为就和时钟频率浮动一样。

在浮动的时钟或发生限制的时候运行 TensorRT 工作负载可能会导致策略选择中更多的不确定性和不稳定的性能测量，因为每个 CUDA 内核可能会以稍微不同的时钟频率运行，具体取决于驱动程序在那一刻提升或限制时钟的频率。另一方面，在锁定时钟的 TensorRT 工作负载中运行可以实现更确定的策略选择和一致的性能测量，但平均性能不如在时钟浮动或在发生限制的情况下将其锁定在最大频率。

关于==是否应该锁定时钟或应该将 GPU 锁定在哪个时钟频率上运行 TensorRT 工作负载没有明确的建议==。这取决于是需要确定性和稳定性能还是最佳平均性能。

#### 13.2.3. GPU 功耗和功率限制

==电源节流 (功率墙)== 发生在 GPU 的平均功耗达到由 `sudo nvidia-smi -pl` 命令设置的功率限制。当其发生时，驱动程序必须将时钟降低到较低的频率，以使平均功耗保持在限制以下。如果在短时间内 (例如20ms内) 进行测量，则不断变化的时钟频率可能会导致性能测量不稳定。

当 GPU 时钟未锁定或锁定在较高频率时，会发生设计上的功率限制，特别是对于功率限制较低的 GPU (例如 NVIDIA T4 和 NVIDIA A2 GPU) ，这是一种自然现象。为避免由功率限制引起的性能变化，您可以将GPU时钟锁定在较低频率，以使性能数字更稳定。但是，此时平均性能数字将低于具有浮动时钟或时钟锁定在较高频率的性能数字，即使发生功率限制。

电源节流的另一个问题是，如果在您的性能基准测试应用的推理之间存在差异，则可能会使性能数字有误差。例如，如果应用程序在每个推理处进行同步，则在推理之间存在空闲时间段。这些差异导致 GPU 平均消耗更少的电力，从而使时钟频率更少地被限制，GPU 可以平均运行在更高的时钟频率上。但是，以这种方式测量的吞吐量数字不准确，因为当 GPU 满载且推理之间没有间隙时，实际时钟频率将更低，并且实际吞吐量将无法达到使用基准应用程序测量的吞吐量数字。

为了避免这种情况，`trtexec` 工具旨在通过几乎不留下 GPU 内核执行之间的间隙来最大化 GPU 执行，以便它可以测量 TensorRT 工作负载的真实吞吐量。因此，如果您在基准测试应用程序和 trtexec 报告之间看到性能差距，请检查功率限制和推理之间的间隙是否是原因。

最后，功耗可能取决于激活值，导致不同输入的不同的性能测量。例如，如果所有网络输入值都设置为零或 NaN，则 GPU 倾向于消耗比输入为正常值时更少的功率，因为 DRAM 和 L2 缓存中的位翻转较少。为避免这种差异，在测量性能时始终使用最能代表实际值分布的输入值。trtexec 工具默认使用随机输入值，但您可以使用 `-–loadInputs` 标志指定输入。有关更多信息，请参见 [trtexec](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec) 部分。

#### 13.2.4. GPU 温度和温度限制

==热节流 (温度墙)== 发生在 GPU 温度达到预定义的阈值 (大多数 GPU 约为 85°C) 时，驱动程序必须将时钟降低到较低的频率，以防止 GPU 过热。您可以通过查看 `nvidia-smi dmon` 命令记录的温度在推理负载运行时逐渐增加，直到达到约 85°C 并开始降低时钟频率来判断这一点。

如果像 Quadro A8000 这样的主动散热型 GPU 发生热限制，则可能是 GPU 上的风扇损坏或有障碍物阻挡了空气流动。

如果像 NVIDIA A10 这样的被动散热型 GPU 发生热限制，则很可能是 GPU 未得到适当的冷却。被动散热型 GPU 需要外部风扇或空调来冷却 GPU，并且空气流必须通过 GPU 才能有效冷却。常见的冷却问题包括将 GPU 安装在不适合 GPU 的服务器中或将错误数量的 GPU 安装到服务器中。在某些情况下，空气流经“简单路径” (即阻力最小的路径) 绕过 GPU 而不是通过它们。如果需要，修复这需要检查服务器中的空气流并安装空气流引导。

请注意，更高的 GPU 温度还会导致电路中更多的漏电电流，从而增加了在特定时钟频率下 GPU 消耗的功率。因此，对于更容易受到功率限制 (例如 NVIDIA T4) 的 GPU，散热不良可能会导致时钟频率降低并出现功率限制，从而导致性能变差，即使 GPU 时钟未被过热限制。

另一方面，只要 GPU 得到适当冷却，环境温度 (即服务器周围环境的温度) 通常不会影响 GPU 性能，除了功率限制较低的 GPU 可能会受到轻微影响。

#### 13.2.5. H2D/D2H 数据传输和 PCIe 带宽

在 dGPU 系统上，通常必须在推理开始之前将输入数据从==主机内存复制到设备内存 (H2D) ==，并且必须在推理后将输出数据从==设备内存复制回主机内存 (D2H) ==。这些 H2D/D2H 数据传输通过 PCIe 总线进行，它们有时会影响推理性能甚至成为性能瓶颈。 H2D/D2H 复制也可以在 Nsight Systems profiles 中看到，表现为 `cudaMemcpy()` 或 `cudaMemcpyAsync()` CUDA API 调用。 

为了实现最大的吞吐量，H2D/D2H 数据传输应与其他推理的 GPU 执行并行运行，以便 GPU 在 H2D/D2H 复制时不会空闲。这可以通过在不同流中并行运行多个推理或通过在与用于 GPU 执行的流不同的流中启用 H2D/D2H 复制并使用 CUDA 事件在流之间进行同步来完成。 trtexec 工具展示了一个后者的实现示例。

当 H2D/D2H 复制与 GPU 执行并行运行时，它们可能会干扰 GPU 执行，特别是如果主机内存是可分页的 (默认情况) 。因此，建议您使用 `cudaHostAlloc()` 或`cudaMallocHost()` CUDA API 为输入和输出数据分配固定的主机内存。

为了检查 PCIe 带宽是否成为性能瓶颈，您可以检查 Nsight Systems profiles，并查看推理查询的 H2D/D2H 复制是否比 GPU 执行部分有更长的延迟。如果 PCIe 带宽成为性能瓶颈，则有几种可能的解决方案。

首先，请检查 GPU 的 PCIe 总线配置是否正确，包括使用哪一代 (例如 Gen3 或 Gen4) 和使用多少条总线 (例如x8或x16) 。接下来，尝试减少必须通过 PCIe 总线传输的数据量。例如，如果输入图像具有高分辨率，并且 H2D 复制成为瓶颈，则可以考虑通过 PCIe 总线传输 JPEG 压缩图像，并在推理工作流之前在 GPU 上解码图像，而不是传输原始像素。最后，您可以考虑使用 NVIDIA GPUDirect 技术直接从/到网络或文件系统加载数据，而无需经过主机内存。

此外，如果您的系统具有 AMD x86_64 CPU，请使用 `numactl --hardware` 命令检查机器的 NUMA (非统一内存访问) 配置。位于两个不同 NUMA 节点上的主机内存和设备内存之间的 PCIe 带宽比位于同一 NUMA 节点上的主机/设备内存之间的带宽要受到更大限制。将主机内存分配到将复制数据的 GPU 所在的 NUMA 节点上。此外，请将触发 H2D/D2H 复制的 CPU 线程固定在该特定 NUMA 节点上。

请注意，在移动平台上，主机和设备共享同一内存，因此如果使用 CUDA API 分配主机内存，并且是锁页的而不是可分页，则不需要进行H2D/D2H数据传输。

默认情况下，`trtexec` 工具测量 H2D/D2H 数据传输的延迟，以告知用户 TensorRT 工作负载是否可能受到 H2D/D2H 复制的瓶颈影响。然而，如果 H2D/D2H 复制影响 GPU 计算时间的稳定性，则可以添加 `-–noDataTransfers` 标志以禁用 H2D/D2H 传输，仅测量GPU执行部分的延迟。

#### 13.2.6. TCC 模式和 WDDM 模式

在 Windows 机器上，有两种驱动程序模式：您可以将 GPU 配置为 ==TCC 模式==和 ==WDDM 模式==。可以通过调用 `sudo nvidia-smi -dm [0|1]` 命令来指定模式，但连接到显示器的 GPU 不应配置为 TCC 模式。有关 TCC 模式的更多信息和限制，请参阅 [TCC 模式文档](https://docs.nvidia.com/nsight-visual-studio-edition/reference/index.html#tesla-compute-cluster)。

在 TCC 模式下，GPU 被配置为专注于计算工作，而 OpenGL 或显视器显示等图形支持被禁用。这是那些运行 TensorRT 推理工作负载的 GPU 的推荐模式。另一方面，WDDM 模式往往会导致 GPU 在使用 TensorRT 运行推理工作负载时有更差和不稳定的性能结果。

这不适用于基于 Linux 的操作系统。

#### 13.2.7. Enqueue-Bound 工作负载和 CUDA Graphs

`IExecutionContext` 的 `enqueue()` 函数是异步的，即在所有 CUDA 内核启动后立即返回，而不等待 CUDA 内核执行完成。但是，在某些情况下，`enqueue()` 时间可能比实际的 GPU 执行时间更长，导致 `enqueue()` 调用的延迟成为性能瓶颈。我们称这种类型的工作负载为 “enqueue-bound”。有两个原因可能导致工作负载被 enqueue-bound。

首先，如果工作负载在计算量方面非常小，例如包含具有小的 I/O 大小的卷积、具有小的 GEMM 大小的矩阵乘法或整个网络中大多数是元素级操作，则工作负载往往会被 enqueue-bound。这是因为==大多数 CUDA 内核需要 CPU 和驱动程序约 5-15 微秒的时间来启动每个内核==，因此如果每个 CUDA 内核执行时间平均只有几微秒长，则内核启动时间成为主要性能瓶颈。

为了解决这个问题，可以尝试增加每个 CUDA 内核的计算量，例如增加批处理大小。或者，您可以使用 [CUDA Graphs](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#cuda-graphs) 将内核启动捕获到 graph 中，并启动 graph 来代替调用 `enqueueV3()`。

其次，如果工作负载包含需要设备同步的操作，例如循环或 if-else 条件，则工作负载自然是 enqueue-bound。在这种情况下，增加批处理大小可能有助于提高吞吐量，而不会显著增加延迟。

在 `trtexec` 中，如果报告的 `Enqueue Time` 接近或长于报告的 `GPU Compute Time`，则可以确定工作负载是 enqueue-bound。在这种情况下，建议您在 `trtexec` 中添加 `–-useCudaGraph` 标志以启用 CUDA graphs，只要工作负载不包含任何同步操作，就可以减少 `Enqueue Time`。

#### 13.2.8. 阻塞同步和旋转等待同步模式

如果使用 `cudaStreamSynchronize()` 或 `cudaEventSynchronize()` 测量性能，则同步开销的变化可能会导致性能测量的变化。本节介绍变化的原因以及如何避免它们。 当调用 `cudaStreamSynchronize()` 时，有两种方式可以使驱动程序等待流的完成。如果使用 `cudaSetDeviceFlags()` 设置了 `cudaDeviceScheduleBlockingSync` 标志，则 `cudaStreamSynchornize()` 使用==阻塞同步==机制。否则，它使用==自旋等待==机制。

类似的想法适用于 CUDA 事件。如果使用 `cudaEventDefault` 标志创建 CUDA 事件，则 `cudaEventSynchronize()` 调用使用自旋等待机制；如果使用 `cudaEventBlockingSync` 标志创建 CUDA 事件，则 `cudaEventSynchronize()` 调用将使用阻塞同步机制。

==当使用阻塞同步模式时，主机线程会让步于另一个线程，直到设备工作完成==。这使得 CPU 可以保持空闲以节省电力或在设备仍在执行时被其他 CPU 工作负载使用。但是，在某些操作系统中，阻塞同步模式往往会导致流/事件同步中的相对不稳定的开销，从而导致延迟测量的变化。

另一方面，==当使用自旋等待模式时，主机线程会不断轮询，直到设备工作完成==。使用自旋等待使延迟测量更加稳定，因为流/事件同步中的开销更短、更稳定，但它会消耗一些 CPU 计算资源，导致 CPU 消耗更多的能量。

因此，如果您想减少 CPU 功耗，或者不希望流/事件同步消耗 CPU 资源 (例如，您正在并行运行其他重型 CPU 工作负载) ，请使用阻塞同步模式。如果您更关心稳定的性能测量，请使用自旋等待模式。

在 `trtexec` 中，默认的同步机制是阻塞同步模式。添加 `-–useSpinWait` 标志以启用使用自旋等待模式进行同步以获得更稳定的延迟测量，代价是更多的 CPU 利用率和功耗。

### 13.3. 优化 TensorRT 性能

以下部分重点介绍了 GPU 上的一般推理流程以及一些提高性能的通用策略。这些想法适用于大多数 CUDA 程序员，但对于来自其他背景的开发人员可能不太明显。

#### 13.3.1. Batching

==最重要的优化是使用批处理尽可能并行地计算尽可能多的结果==。在 TensorRT 中，批处理是可以统一处理的一组输入。批次中的每个实例具有相同的形状，并以完全相同的方式流经网络。因此，每个实例都可以轻松地并行计算。 

网络的每个层都需要一定量的开销和同步来计算前向推理。通过并行计算更多结果，可以更有效地摊销这些开销。此外，许多层性能受限于输入中最小的维度。如果批处理大小为 1 或较小，则此大小通常会成为限制性能的维度。例如，具有 `V` 个输入和 `K` 个输出的全连接层可以针对一个批次实例实现为 `1xV` 矩阵与 `VxK` 权重矩阵的矩阵乘法。如果 `N` 个实例被批量处理，则这将变为 `NxV` 乘以 `VxK` 矩阵。向量-矩阵乘法器变成了矩阵-矩阵乘法器，这样更高效。

在 GPU 上，较大的批处理大小几乎总是更高效的。极大的批次，例如 `N > 2^16`，有时可能需要扩展索引计算，因此应尽可能避免。但==通常，增加批处理大小可以提高总吞吐量==。此外，当网络包含矩阵乘法层或全连接层时，批处理大小为 32 的倍数往往对于 FP16 和 INT8 推理有最好的性能，因为如果硬件支持，可以利用 Tensor Cores。

在 NVIDIA Ada Lovelace GPU 或更高版本的 GPU 上，==如果较小的批处理大小恰好有助于 GPU 在 L2 缓存中缓存输入/输出值，则减小批处理大小可能会显着提高吞吐量==。因此，请尝试各种批处理大小以获得最佳性能的批处理大小。

有时由于应用程序的组织，无法对推理工作进行批处理。在一些常见的应用程序中，例如为每个请求执行推理的服务端，可以实现取巧的批量处理。对于每个传入请求，等待时间 `T`。如果在此期间有其他请求，则将它们一起批处理。否则，继续进行单个实例推理。这种策略为每个请求添加固定延迟，但可以将系统的最大吞吐量提高几个数量级。

[NVIDIA Triton 推理服务](https://developer.nvidia.com/nvidia-triton-inference-server)提供了一种简单的方法来使用 TensorRT 引擎启用动态批处理。

使用批处理：

如果在创建网络时使用显式批处理模式，则批处理维度是张量维度的一部分，您可以通过添加优化配置文件来指定批处理大小的范围和要优化引擎的批处理大小。更多细节请参考[动态形状](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes)部分。

如果在创建网络时使用隐式批处理模式，则 `IExecutionContext::execute` (Python 中的 `IExecutionContext.execute`) 和 `IExecutionContext::enqueue` (Python 中的 `IExecutionContext.execute_async`) 方法需要一个批处理大小参数。在构建优化网络时，还应使用 `IBuilder::setMaxBatchSize` (Python 中的 `Builder.max_batch_size`) 设置最大批处理大小。在调用 `IExecutionContext::execute` 或 `enqueue` 时，作为 `bindings` 参数传递的 bindings 是按每个张量而不是按每个实例来组织的。换句话说，一个输入实例的数据不会被一起分组到内存的一块连续区域中。相反，每个张量 bindings 都是该张量的一个实例数据数组。

另一个顾虑是构建优化网络会是针对给定的最大批处理大小进行优化。==最终结果将针对最大批处理大小进行调整，但同样可以正确地处理任何较小的批处理大小==。可以运行多个构建操作以创建多个针对不同批处理大小进行优化的引擎，然后根据运行时实际批处理大小选择要使用的引擎。

#### 13.3.2. Within-Inference Multi-Streaming

一般来说，CUDA 编程流是组织异步工作的一种方式。放入流中的异步命令保证按顺序运行，但相对于其他流的可能无序执行。特别地，两个流中的异步命令可能被调度为并发运行 (受硬件限制)。

在 TensorRT 和推理的上下文中，优化后的最终网络的每个层都需要在 GPU 上工作。但是，并非所有层都能够充分利用硬件的计算能力。在单独的流中的调度请求允许在硬件可用时立即安排工作，而无需进行不必要的同步。即使只有一些层可以重叠，整体性能也会提高。

从 TensorRT 8.6 开始，您可以使用 `IBuilderConfig::setMaxAuxStreams()` API 来设置 TensorRT 允许使用的最大辅助流数，以并行运行多个层。辅助流与 `enqueueV3()` 调用中提供的“主流”相对应，如果启用，则 TensorRT 将在辅助流上并行运行一些层 (与主流上运行的层并行运行)。

例如，要在总共最多八个流 (即七个辅助流和一个主流) 上运行推理：

- **C++**

  ```c++
  config->setMaxAuxStreams(7);
  ```

- **Python**

  ```python
  config.max_aux_streams = 7
  ```

请注意，这仅设置了辅助流的最大数量，但是，如果 TensorRT 确定使用更多流不会有所帮助，则 TensorRT 可能会使用比此数量更少的辅助流。

要获取 TensorRT 为引擎使用的实际辅助流数，请运行：

- **C++**

  ```c++
  int32_t nbAuxStreams = engine->getNbAuxStreams();
  ```

- **Python**

  ```python
  num_aux_streams = engine.num_aux_streams
  ```

当从引擎创建执行一个上下文时，TensorRT 会自动创建运行推理所需的辅助流。但是，您也可以指定您希望 TensorRT 使用的辅助流：

- **C++**

  ```c++
  int32_t nbAuxStreams = engine->getNbAuxStreams();
  std::vector<cudaStream_t> streams(nbAuxStreams);
  for (int32_t i = 0; i < nbAuxStreams; ++i)
  {
      cudaStreamCreate(&streams[i]);
  }
  context->setAuxStreams(streams.data(), nbAuxStreams);
  ```

- **Python**

  ```c++
  from cuda import cudart
  num_aux_streams = engine.num_aux_streams
  streams = []
  for i in range(num_aux_streams):
      err, stream = cudart.cudaStreamCreate()
      streams.append(stream)
  context.set_aux_streams(streams)
  ```

TensorRT 将始终在 `enqueueV3()` 调用中提供的主流和辅助流之间插入事件同步： 

- 在 `enqueueV3()` 调用的开始处，TensorRT 将确保所有辅助流等待主流上的活动。
- 在 `enqueueV3() `调用的结束处，TensorRT 将确保主流等待所有辅助流上的活动。 


请注意，启用辅助流可能会增加内存消耗，因为某些激活的缓冲区将无法再被重复使用。

#### 13.3.3. Cross-Inference Multi-Streaming

除了推理内的流之外，您还可以在多个执行上下文之间启用流。例如，您可以使用多个优化配置构建引擎，并为每个配置创建一个执行上下文。然后，在不同的流上调用执行上下文的 `enqueueV3()` 函数以运行它们。

运行多个并发流通常会导致几个流同时共享计算资源。这意味着在推理期间，网络可用的计算资源可能比 TensorRT 引擎优化时少。资源可用性的差异可能会导致 TensorRT 选择对实际运行时条件欠佳的内核。为了减轻这种影响，您可以在引擎创建期间限制可用的计算资源量，以更接近实际运行时条件。这种方法通常以延迟为代价来提高吞吐量。更多信息请参见[有限计算资源](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#limit-compute-resources)。

也可以将多个主机线程与流一起使用。常见模式是将传入的请求分派到等待工作线程池中。在这种情况下，工作线程池中每个线程将有一个执行上下文和 CUDA 流。当工作变得可用时每个线程将在其自己的流中请求工作。每个线程将与其流同步以等待结果，而不会阻塞其他工作线程。

#### 13.3.4. CUDA Graphs

CUDA Graphs 是一种表示内核序列 (或图形) 的方式，这种方式允许 CUDA 优化其调度。当您的应用程序性能对入队内核所需的 CPU 时间敏感时，这可能特别有用。

TensorRT 的 `enqueuev3()` 方法支持不需要在管道中间进行 CPU 交互的模型的 CUDA graph 捕获。例如：

- **C++**

  ```c++
  // 在输入形状更改后调用 enqueueV3() 一次来更新内部状态
  context->enqueueV3(stream);
  
  // 捕获一个 CUDA graph 实例
  cudaGraph_t graph;
  cudaGraphExec_t instance;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  context->enqueueV3(stream);
  cudaStreamEndCapture(stream, &graph);
  cudaGraphInstantiate(&instance, graph, 0);
  
  // 要运行推理，启动 graph，而不是调用 enqueueV3()
  for (int i = 0; i < iterations; ++i) { 
      cudaGraphLaunch(instance, stream);
      cudaStreamSynchronize(stream);
  }
  ```

- **Python**

  ```python
  from cuda import cudart
  err, stream = cudart.cudaStreamCreate()
  
  # 在输入形状更改后调用 execute_async_v3() 一次来更新内部状态
  context.execute_async_v3(stream);
  
  # 捕获一个 CUDA graph 实例
  cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureModeGlobal)
  context.execute_async_v3(stream)
  err, graph = cudart.cudaStreamEndCapture(stream)
  err, instance = cudart.cudaGraphInstantiate(graph, 0)
  
  # 要运行推理，启动 graph，而不是调用 enqueueV3()
  for i in range(iterations):
      cudart.cudaGraphLaunch(instance, stream)
      cudart.cudaStreamSynchronize(stream)
  ```

具有循环或条件的模型不支持 graph。在这种情况下，`cudaStreamEndCapture()` 将返回 `cudaErrorStreamCapture*` 错误，表示 graph 捕获失败，但上下文仍然可以用于不使用 CUDA graph 的正常推理。

在捕获 graph 时，重要的是要考虑存在动态形状时使用的两阶段执行策略。

1. 更新模型的内部状态以考虑输入大小的任何更改。 
2. 将工作流化放到 GPU 

对于在构建时输入大小固定的模型，第一阶段不需要每次调用工作。反之，如果自上次调用后输入大小发生了更改，则可能需要一些工作来更新派生属性。

第一阶段的工作不是设计为被捕获的，即使捕获成功也可能会增加模型执行时间。因此，在更改输入形状或形状张量的值后，请在捕获 graph 之前调用 `enqueueV3()` 一次以刷新延迟更新。

使用 TensorRT 捕获的 graph 是特定于它们被捕获的输入大小的，也特定于执行上下文的状态。修改捕获 graph 的上下文将导致执行 graph 时出现未定义行为。特别是，如果应用程序使用 `createExecutionContextWithoutDeviceMemory()` 提供自己的内存给激活，则内存地址也将作为 graph 的一部分被捕获。绑定位置也作为 graph 的一部分被捕获。

因此，最佳做法是每个捕获的 graph 使用一个执行上下文，并使用 `createExecutionContextWithoutDeviceMemory()` 在上下文之间共享内存。

`trtexec` 允许您检查构建的 TensorRT 引擎是否与 CUDA graph 捕获兼容。更多信息请参见 `trtexec` 部分。

#### 13.3.5. 启用融合

##### 13.3.5.1. 层融合

TensorRT 在构建阶段尝试在网络中执行许多不同类型的优化。在第一阶段中，尽可能地将层融合在一起。融合将网络转换为更简单的形式，但保持相同的整体行为。在内部，许多层的实现具有额外的参数和选项，这些参数和选项在创建网络时不能直接访问。相反，融合优化步骤检测支持的操作模式，并用内部选项集将多个层融合成一个层。

考虑卷积后接 ReLU 激活的常见情况。要创建具有这些操作的网络，使用 `addConvolution` 添加一个卷积层，然后使用 `ActivationType` 为 `kRELU` 的 `addActivation` 添加一个激活层。未经优化的 graph 将包含独立的卷积和激活层。卷积的内部实现支持直接从卷积核输出上一步计算 ReLU 函数而无需进行第二个内核调用。融合优化步骤将检测到卷积后接着 ReLU。验证实现是否支持这些操作，然后将它们融合成一个层。

要调查哪些融合已发生或未发生，构建器将其操作记录到在构造期间提供的日志对象中。优化步骤位于 `kINFO`日志级别。要查看这些消息，请确保在 `ILogger` 回调中记录它们。

通常通过创建一个新层来处理融合，该新层包含已融合层名称的名称。例如，在 MNIST 中，名为 `ip1` 的全连接层 (内积) 与名为 `relu1` 的 ReLU 激活层融合以创建一个名为 `ip1 + relu1` 的新层。

##### 13.3.5.2. 融合的类型

以下列表描述了支持的融合类型。

- **ReLU激活**：

  一个 ReLU 的激活层后接着一个 ReLU 的激活将被替换为一个单激活层。

- **卷积和 ReLU 激活**

  卷积层可以是任何类型，值没有限制。激活层必须是 ReLU 类型。

- **卷积和 GELU 激活**

  输入和输出的精度应相同；两者都是 FP16 或 INT8。激活层必须是 GELU 类型。TensorRT 应在 NVIDIA Turing 或更高设备上运行，使用 CUDA 版本10.0 或更高版本。

- **卷积和 Clip 激活**

  卷积层可以是任何类型，值没有限制。激活层必须是 Clip 类型。

- **scale 和激活**

  Scale 层后跟激活层可以融合成单个激活层。 

- **卷积和逐元素运算 **

  在逐元素层中，卷积层后跟简单的 sum、min 或 max 可以融合到卷积层中。除非广播是在批量大小维度上，否则 sum 不能使用广播。

- **填充和卷积/反卷积**

  如果所有填充大小都是非负数，则填充后跟卷积或反卷积可以融合成单个卷积/反卷积层。 

- **Shuffle 和 Reduce **

  没有 reshape 的 Shuffle 层后跟 Reduce层可以融合成单个 Reduce 层。Shuffle 层可以执行 permute，但不能执行任何 reshape 操作。Reduce 层必须具有一组 `keepDimensions` 的维度。

- 





































## 14. 故障排除

以下部分帮助回答有关典型用例的最常见问题。

### 14.1. FAQs

**问：如何创建一个针对不同 batch 大小进行优化的engine ?**

答：虽然 TensorRT 允许针对给定批量大小优化的 engine 在任意更小尺寸下运行，但这些较小尺寸的性能无法得到很好的优化。要针对多个不同的批量大小进行优化，请在 `OptProfilerSelector::kOPT` 指定的维度上创建优化配置文件。



**问：校准表在不同 TensorRT 版本间可移植吗?**

答：不可以。内部实现会不断优化，不同版本间可能会改变。因此，不能保证校准表与不同的 TensorRT 版本二进制兼容。当使用新 TensorRT 版本时，应用程序必须构建新的 `INT8` 校准表。



**问： engines 在不同 TensorRT 版本间可移植吗?**

答：默认情况下不可以。请参考[版本兼容性](#6.1. Version Compatibility)了解如何配置 engines 实现向前兼容。



**问：如何选择最佳的工作空间大小?**

答：一些 TensorRT 算法需要在 GPU 上额外的工作空间。`IBuilderConfig::setMemoryPoolLimit()` 方法控制可以分配的最大工作空间量，并防止需要更多工作空间的算法被 `builder` 考虑。在运行时，当创建 `IExecutionContext` 时，空间会自动分配。即使在 `IBuilderConfig::setMemoryPoolLimit()` 中设置的量更大，分配的量也不会超过所需。因此，应用程序应该尽可能为 TensorRT `builder` 提供尽可能多的工作空间；在运行时，TensorRT 只分配所需量，通常更少。



**问：如何在多个 GPU 上使用 TensorRT ?**

答：每个 `ICudaEngine` 对象在实例化时都会绑定到特定 GPU，无论是通过 `builder` 还是反序列化。要选择 GPU，在调用 builder 或反序列化 engine 之前使用 `cudaSetDevice()`。每个 `IExecutionContext` 与创建它的 engine 绑定在同一个 GPU 上。调用 `execute()` 或 `enqueue()` 时，如果需要，通过调用 `cudaSetDevice()` 确保线程与正确的设备相关联。

> 在内存向 GPU 传递数据时，建议再次指定设备，否则这个复制线程可能不知道到底向哪个 GPU 传递
>
> 多卡多线程需要在创建实例和上下文、推理、拷贝数据前指定设备，否则可能会遭遇错误：**Cask Error in checkCaskExecError<false>: 7 (Cask Convolution execution) **
>
> 参考：
>
> 1. https://github.com/NVIDIA/TensorRT/issues/219#issuecomment-559249117
> 2. https://github.com/NVIDIA/TensorRT/issues/301#issuecomment-687630290
> 3. https://blog.csdn.net/qq_36184671/article/details/114115249



**问：如何从库文件中获取 TensorRT 的版本?**

答：符号表中有一个名为 `tensorrt_version_#_#_#_#` 的符号包含了 TensorRT 的版本号。在 Linux 上获取这个符号的一种方法是使用 `nm` 命令，如下面的例子:

```bash
nm -D libnvinfer.so.* | grep tensorrt_version
```

输出类似下面

```
00000000abcd1234 B tensorrt_version_#_#_#_#
```



**问：如果我的网络生成了错误的结果，我该如何排查?**

答：您的网络生成错误结果有几个可能的原因。这里有一些可以帮助诊断问题的故障排除方法:

- 打开日志流中的 `VERBOSE` 级别消息，并检查 TensorRT 的报告。

- 检查您的输入预处理是否准确生成网络所需的输入格式。

- 如果您使用的是降低的精度，请用 `FP32` 运行网络。如果它生成了正确的结果，则较低的精度可能导致网络的动态范围不足。

- 尝试将网络中的中间张量标记为输出，并验证它们是否与您的预期匹配。

  注意：将张量标记为输出可能会抑制优化，从而改变结果。

您可以使用 [NVIDIA Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy) 来协助调试和诊断。



**问：如何 在TensorRT 中实现批量标准化?**

答：批量标准化可以在 TensorRT 中使用一系列 `IElementWiseLayer` 来实现。具体而言:

    adjustedScale = scale / sqrt(variance + epsilon)
    batchNorm = (input + bias - (adjustedScale * mean)) * adjustedScale



**问：为什么使用 DLA 时我的网络运行速度比不使用 DLA 时慢?**

答：DLA 被设计为最大化能效。根据 DLA 支持的特性和 GPU 支持的特性，任一实现都可能更高效。使用哪个实现取决于您的延迟或吞吐量要求以及功耗预算。由于所有 DLA 引擎独立于 GPU 和彼此独立，您也可以同时使用两种实现以进一步提高网络的吞吐量。



**问：TensorRT 目前是否支持 INT4 或 INT16 量化?**

答：TensorRT 目前还不支持 `INT4` 或 `INT16` 量化。



**问：TensorRT 的 UFF 解析器什么时候会支持我的网络所需的 XYZ 层?**

答：UFF 已被废弃。我们建议用户将工作流切换到 ONNX。TensorRT ONNX 解析器是一个开源项目。



**问：我可以使用多个 TensorRT builder 在不同目标上进行编译吗?**

答：TensorRT 假设它所在的构建设备的所有资源在优化过程中都是可用的。并发使用多个 TensorRT `builder` (例如，多个 `trtexec` 实例) 在不同目标 (DLA0、DLA1 和 GPU) 上进行编译可能会过度订阅系统资源，导致未定义的行为 (即低效计划、构建失败或系统不稳定)。

建议您使用 `trtexec` 和 `--saveEngine` 参数分别为不同目标 (DLA 和 GPU) 编译并保存其 plan 文件。然后，这些 plan 文件可以用于加载 (使用带有 `--loadEngine` 参数的 `trtexec`) 并在各自的目标 (DLA0、DLA1、GPU) 上提交多个推理作业。这种两步过程可以减轻构建阶段的资源过度订阅，同时允许执行 plan 文件而不受 `builder` 干扰。



**问：哪些层被 Tensor Core 加速?**

答：大多数算力密集的操作将由 tensor cores 加速—卷积、反卷积、全连接和矩阵乘法。在某些情况下，特别是对于小的通道数或小的组尺寸 (group sizes)，可能会选择另一种实现而不是 tensor core 实现，因为它可能更快。



**问：为什么观察到 reformatting 层，尽管没有警告消息“没有实现遵守无 reformatting 规则”?**

答：无 reformatting 网络输入输出并不意味着整个网络中不插入 reformatting 层。仅意味着网络输入输出张量有可能不需要 reformatting 层。换句话说，TensorRT 可以为内部张量插入 reformatting 层以提高性能。



### 14.2. 理解错误消息

如果在执行过程中遇到错误，TensorRT 会报告一条错误消息，旨在帮助调试问题。以下部分讨论了开发人员可能遇到的一些常见错误消息。

#### UFF 解析器错误消息

下表捕获了常见的 UFF 解析器错误消息。

| 错误信息                                                     | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `The input to the Scale Layer is required to have a minimum of 3 dimensions.` | 由于输入尺寸不正确，可能会出现此错误消息。在 UFF 中，输入维度应始终使用规范中未包含的隐式批量维度来指定。 |
| `Invalid scale mode, nbWeights: <X>                          | 同上                                                         |
| `kernel weights has count <X> but <Y> was expected`          | 同上                                                         |
| `<NODE> Axis node has op <OP>, expected Const. The axis must be specified as a Const node.` | 如错误消息所示，轴必须是构建时间常量，以便 UFF 正确解析节点。 |

#### ONNX 解析器错误消息

下表捕获了常见的 ONNX 解析器错误消息。有关特定 ONNX 节点支持的更多信息，请参阅[运营商支持](https://github.com/onnx/onnx/blob/main/docs/Operators.md)文档。

| 错误信息                                                     | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `<X> must be an initializer!`                                | 这些错误消息意味着 ONNX 节点输入张量在 TensorRT 中预期是初始化器。一个可能的解决方法是使用 TensorRT 的 Polygraphy 工具对模型运行常量折叠：`polygraphy surgeon sanitize model.onnx --fold-constants --output model_folded.onnx` |
| `!inputs.at(X).is_weights()`                                 | 同上                                                         |
| `getPluginCreator() could not find Plugin <operator name> version 1` | 这是一个错误，表明 ONNX 解析器没有为特定运算符定义的导入函数，并且在加载的注册表中没有找到该运算符的相应插件。 |

#### TensorRT 核心库错误消息

下表捕获了常见的 TensorRT 核心库错误消息。

|                | 错误信息                                                     | 描述                                                         |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **安装错误**   | `Cuda initialization failure with error <code>. Please check cuda installation: `http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html. | 如果 CUDA 或 NVIDIA 驱动程序安装损坏，可能会出现此错误消息。请参阅该 URL，了解有关在操作系统上安装 CUDA 和 NVIDIA 驱动程序的说明。 |
| **生成器错误** | `Internal error: could not find any implementation for node <name>. Try increasing the workspace size with IBuilderConfig::setMemoryPoolLimit().` | 出现此错误消息的原因是网络中的给定节点没有可以在给定工作空间大小下运行的层实现。发生这种情况通常是因为工作空间大小不足，但也可能表明存在错误。如果按照建议增加工作区大小没有帮助，请报告错误（请参阅[报告 TensorRT 问题](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#reporting-issues)）。 |
|                | `<layer-name>: (kernel|bias) weights has non-zero count but null ` `values <layer-name>: (kernel|bias) weights has zero count but non-null values` | 当传递给构建器的权重数据结构中的值和计数字段不匹配时，会出现此错误消息。如果计数是 0，那么 `values` 字段必须包含空指针；否则，计数必须非零，并且值必须包含非空指针。 |

