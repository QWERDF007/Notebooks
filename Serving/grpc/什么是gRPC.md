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