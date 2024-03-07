# 什么是 gRPC？

## gRPC 介绍

gRPC 可以使用 protocol buffers (协议缓冲区?) 作为它的接口定义语言 (IDL) 和它的底层信息交换格式。

### 概述

在 gRPC 中，客户端应用程序可以直接调用不同机器上的服务端应用程序的方法，就好像它是本地对象一样，这使得创建分布式应用和服务更加容易。和许多 RPC 系统一样，gRPC 是基于定义服务的思想，指定可以通过参数和返回类型远程调用的方法。在服务端，服务器实现这个接口，并运行一个 gRPC 服务来处理客户端调用。在客户端，客户端有一个存根 (在模型语言中称为客户端) 提供与服务端相同的方法。

<img src="./assets/landing-2.svg" style="zoom:100%">

gRPC 客户端和服务器可以在各种环境中运行和彼此通信——从谷歌内部服务器到你的桌面电脑，并且可以用任何 gRPC 支持的语言编写。例如，你可以很容易地用 Java 创建服务器，用 Go、Python 或 Ruby 创建客户端。另外，最新的 Google APIs 接口将有 gRPC 版，让你轻松地将谷歌功能构建到你的应用中。

### 使用 Protocol Buffers

默认情况下，gRPC 使用 Protocol Buffers，谷歌用于序列化数据的成熟的开源方案 (经管其他数据格式例如 Json 也能做到)。下面是它如何工作的一个快速介绍。

使用 protocol buffers 的第一步是在 proto 文件中定义你需要序列化的数据的结构：这是一个扩展名为 `.proto` 的普通文本文件。Protocol buffer 数据被构造为消息，其中每个消息是信息的一个小的逻辑记录，包含一系列名称值对，叫做字段。下面是一个简单的例子

```protobuf
message Person{
  string name = 1;
  int32 id = 2;
  bool has_ponycopter = 3;
}
```

然后，一旦你指定了你的数据结构，就可以用 protocol buffer 编译器 `protoc` 从你的 proto 定义中生成你首选语言的数据访问类。它们为每个字段提供了简单的访问器，比如 `name()` 和 `set_name()`，以及将整个结构序列化/解析为原始字节的方法。例如，你选用的语言是 C++，在上面的示例中运行编译器将生成应该名为 `Person` 的类。你可以在你的应用中使用这个类来填充、序列化和检索 `Person` protocol buffers 消息。

你可以在普通的 proto 文件中定义 gRPC 服务，使用 RPC 方法参数和指定为 protocol buffers 消息的返回类型：

```protobuf
// The greeter service definition.
service Greeter {
  // Sends a greeting
  rpc SayHello (HelloRequest) returns (HelloReply) {}
}

// The request message containing the user's name.
message HelloRequest {
  string name = 1;
}

// The response message containing the greetings
message HelloReply {
  string message = 1;
}
```

gRPC 使用带有特殊插件 `protoc` 从你的 proto 文件中生成代码：你得到生成的 gRPC 客户端和服务器代码，和用于填充、序列化和检索你的消息类型的常规 protocol buffer 代码。下面给出一个示例。

要了解更多关于 protocol buffers，包括如何在您选择的语言中安装带有 gRPC 插件的 `protoc`，请参阅 [protocol buffers documentation](https://developers.google.com/protocol-buffers/docs/overview)。

### Protocol buffer 版本

虽然 protocol buffers 已供开源用户使用一段时间了，但是这个站点的大部分例子使用 protocol buffers version 3 (proto3)，它具有稍微简化的语法，一些有用的新特性，并支持更多的语言。Proto3 目前支持 Java、C++、Dart、Python、Object-C、C#、lite-runtime (Android Java)、Ruby 和来自 protocol buffers Github repo 的 JavaScript，以及来自 golang/protobuf 官方包的 Go 语言生成器，更多语言正在开发中。你可以在 [proto3 language guide](https://developers.google.com/protocol-buffers/docs/proto3) 和每种语言的[参考文档](https://developers.google.com/protocol-buffers/docs/reference/overview)中找到更多信息。参考文档还包含 `.profo` 文件格式的[正式规范](https://developers.google.com/protocol-buffers/docs/reference/proto3-spec)。

总体来说，虽然你可以使用 proto2 (当前的默认的 protocol buffers 版本)，但强烈推荐你使用带有 gRPC 的 proto3，因为它能让你使用 gRPC 支持的语言的所有范围，以及避免 proto2 客户端与 proto3 服务器通信时的兼容性问题，反之亦然。

## 核心概念、架构和生命周期

介绍 gRPC 关键概念，概述 gRPC 架构和生命周期。

不熟悉 gRPC？首先阅读 [gRPC 介绍](#gRPC 介绍)。关于特定语言的细节，请参阅你所选择的语言的快速入门，教程和参考文档。

### 概述

#### 服务定义

像许多 RPC 系统一样，gRPC 是基于定义服务的思想，指定可以通过参数和返回类型远程调用的方法。默认情况下，gRPC 使用 [protocol buffers]() 作为接口定义语言 (IDL) 来描述服务接口和有效负载消息的结构。如果需要，也可以使用其他代替方法。

```protobuf
service HelloService {
  rpc SayHello (HelloRequest) returns (HelloResponse);
}

message HelloRequest {
  string greeting = 1;
}

message HelloResponse {
  string reply = 1;
}
```

gRPC 允许你定义四种服务方法：

- Unary RPCs，客户端向服务端发送单个请求并获得单个响应，就像普通函数调用一样。

  ```protobuf
  rpc SayHello(HelloRequest) returns (HelloResponse);
  ```

- 服务端流 RPC，客户端像服务端发送一个请求，并获得一个流来读取消息序列。客户端从返回的流中读取，直到没有更多消息。gRPC 保证在一个独立 RPC 调用中的消息的顺序。

  ```protobuf
  rpc LotsOfReplies(HelloRequest) returns (stream HelloResponse);
  ```

- 客户端流 RPC，客户端写入消息序列，然后再次使用提供的流将它们发送给服务端。一旦客户端完成写入消息序列，它就会等待服务端读取消息并返回响应。同样，gRPC 保证在一个独立 RPC 调用中的消息顺序。

  ```protobuf
  rpc LotsOfGreetings(stream HelloRequest) returns (HelloResponse);
  ```

- 双向流 RPC，两边都使用一个读写流发送消息序列。两个流独立运作，因此客户端和服务端可以按任意顺序读写：例如，服务端可以在写响应之前等待接收完所有客户端消息，或者交替读一条消息然后写一条消息，或者其他一些读写组合。每个流中的消息的顺序被保留。

  ```protobuf
  rpc BidiHello(stream HelloRequest) returns (stream HelloResponse);
  ```

在下面的 RPC 生命周期部分中，你将了解更多有关不同类型的 RPC 信息。

#### 使用 API

从 `.proto` 文件中的服务定义开始，gRPC 提供生成客户端和服务端代码的 protocol buffers 编译器插件。gRPC 用户通常在客户端调用这些 APIs，并在服务端实现相应的 API。

- 在服务端，服务端实现以 service 定义的方法，并运行 gRPC 服务来处理客户端调用。gRPC 的基础架构解码即将到来的请求，执行服务方法，并编码服务响应。
- 在客户端，客户端有一个称为存根的本地对象 (对于某些语言，首选术语是客户端) 实现了与服务相同的方法。然后客户端可以在本地对象上调用这些方法，将调用的参数包装在合适的 protocol buffer 消息类型中——gRPC 负责像服务端发送请求，并返回服务端的 protocol buffer 响应。

#### 同步 vs. 异步

在服务端响应到达之前的将一直阻塞的同步 RPC 调用是 RPC 所期望的过程调用的最接近的抽象。另一方面，网络天生就是异步的，在很多情况下，能够在不阻塞当前线程的情况下启动 RPCs 是很有用的。

大多数语言中的 gRPC 编程 API 都有同步和异步两种风格。你可以在每种语言的教程和参考文档中找到更多信息。

### RPC 生命周期

在本节中，将进一步了解当 gRPC 客户端调用 gRPC 服务方法时发送了什么。有关完整的实现细节，请参阅语言特定页面。

#### Unary RPC

首先考虑最简单的 RPC 类型，客户端发送单个请求并获得单个响应。

1. 一旦客户端调用存根的方法，服务端就会被告知 RPC 已经被调用，该调用带有客户端的元数据，方法名和指定的截止日期 (如果适用)。
2. 服务端可以直接返回它自己的初始元数据 (必须在响应之前发送)，或者等待客户端请求消息。谁先发送由应用程序指定。
3. 一旦服务端获得了客户端的请求消息，它就会执行创建和填充响应所需的全部工作。然后将响应 (如果成功) 连同状态详细信息 (状态码和可选的状态消息) 和可选的跟踪元数据返回给客户端。
4. 如果响应状态为 OK，则客户端将得到响应，从而完成客户端上的调用。

#### 服务端流 RPC

服务端流 RPC 类似于 Unary RPC，除了服务端返回的是一个响应客户端请求的消息流。在发送完所有消息之后，服务端的状态详细信息 (状态码和可选的状态消息) 和可选的跟踪元数据被发送给客户端。这就完成了服务端的处理。客户端在获取完所有服务端的消息后完成。

#### 客户端流 RPC

客户端流 RPC 类似于 Unary RPC，除了客户端向服务端发送消息流而不是单个消息。服务端用一条消息 (连同它的状态详细信息和可选的跟踪元数据) 响应，通常但不一定是在它收到所有的客户端消息之后。

#### 双向流 RPC

在双向流 RPC 中，调用是以客户端调用方法，并且服务端接收到客户端的元数据，方法名和截止日期开始的。服务端可以选择返回它的初始元数据或者等待客户端开启消息流。

客户端和服务端的流处理是特定于应用程序的。因为这两个流是独立的，客户端和服务端能以任意顺序读写消息。例如，服务端可以在写消息之前等待接收完客户端所有的消息，或者服务端和客户端可以玩”乒乓球“——服务端收到一个请求，然后发回一个响应，客户端根据这个响应发送另外一个请求，等等。

#### 截止期限/超时

gRPC 允许客户端指定在 RPC 因 `DEADLINE_EXCEEDED` 错误而终止之前愿意等待多长时间。在服务端，服务器可以查询特定的 RPC 是否超时，或者还剩下多少时间来完成 RPC。

指定截止期限或超时是特定于语言的：一些语言的 API 使用超时 (时间段)，而一些语言的 API 使用截止日期 (一个固定的时间点)，可能有也可能没有默认的截止日期。

#### RPC 终止

在 gRPC，客户端和服务端对调用的成功性做出独立的和本地的判断，并且它们的结论可能不匹配。这意味着，例如，你有一个 RPC，它在服务端成功完成 ("发送了所有响应")，但在客户端失败 ("响应在截止日期后到达")。服务端也可以在客户端发送完所有请求之前决定是否完成。

#### 取消 RPC

客户端或服务端可以在任何时候取消 RPC。取消会立刻终止 RPC，这样就不用做进一步的工作了。

警告：取消之前所作的更改不会回滚。

#### 元数据

元数据是以键值对列表的形式表示的关于特定 RPC 调用的信息 (比如身份验证细节)，其中键是字符串，值通常是字符串，但也可以是二进制数据。元数据对于 gRPC 本身是不透明的——它让客户端提供调用的相关信息给客户端，反之亦然。

访问元数据依赖于语言。

#### 通道

一个 gRPC 通道提供到指定主机和端口的 gRPC 服务的连接。它在创建客户端存根时使用。客户端可以指定通道参数来修改 gRPC 的默认行为，比如打开或关闭消息压缩。通道有状态，包括 `connected` 和 `idle`。

gRPC 如何关闭通道取决于语言。一些语言还允许查询通道状态。



<!-- 完成标志, 看不到, 请忽略! -->
