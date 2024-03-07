# [Creating C++ Plugins for QML](https://doc.qt.io/qt-6/qtqml-modules-cppplugins.html)

## Creating a Plugin

QML 引擎可以加载 C++ 插件，这些插件通常在 QML 扩展模块中提供，并且可以在导入模块的 QML 文档中的客户端提供类型。模块至少需要一个已注册的类型才能被视为有效。

[QQmlEngineExtensionPlugin](https://doc.qt.io/qt-6/qqmlengineextensionplugin.html) 是一种插件接口，可让您创建可以动态加载到 QML 应用程序中的 QML 扩展。这些扩展允许自定义的 QML 类型可用于 QML 引擎。

要编写QML扩展插件：

1. 子类化 `QQmlEngineExtensionPlugin` 并使用 `Q_PLUGIN_METADATA()` 宏将插件注册到 Qt 元对象系统。

2. 使用 `QML_ELEMENT` 和 `QML_NAMED_ELEMENT()` 宏声明QML类型。

3. 配置您的构建文件。

   CMake:

   ```cmake
   qt_add_qml_module(<target>
       URI <my.import.name>
       VERSION 1.0
       QML_FILES <app.qml>
       NO_RESOURCE_TARGET_PATH
   )
   ```

   qmake:

   ```qmake
   CONFIG += qmltypes
   QML_IMPORT_NAME = <my.import.name>
   QML_IMPORT_MAJOR_VERSION = <version>
   ```

4. 如果是使用 qmake，要创建一个 [qmldir](https://doc.qt.io/qt-6/qtqml-modules-qmldir.html) 文件来描述这个插件。注意 CMake 将会自动生成这个 qmldir 文件。

QML 扩展插件有 application-specific 和 library-like 插件。库插件应仅限于注册类型，因为对引擎的根上下文的任何操作可能会导致库用户代码中的冲突或其他问题。

> 注意：使用 CMake `qt_add_qml_module` API 时，将自动生成一个插件。它将负责类型注册。如果您有特殊要求（例如注册自定义图像提供程序），则只需要编写自定义插件。在这种情况下，将`NO_GENERATE_PLUGIN_SOURCE` 传递给 `qt_add_qml_module` 的调用以禁用默认插件的生成。

链接器可能会错误地将生成的类型注册函数作为优化删除。您可以通过在代码中的某个位置声明一个指向该函数的 人造的易失性指针 (volatile pointer) 来防止这种情况。如果您的模块名为 "my.module"，则可以在全局范围内添加前向声明：

```c++
void qml_register_types_my_module();
```

然后在与注册相同的二进制文件的任何函数的实现中添加以下代码片段：

```c++
volatile auto registration = &qml_register_types_my_module;
Q_UNUSED(registration);
```

## Reference

- [Writing QML Extensions with C++](<./Writing QML Extensions with C++.md>) - 包含有关创建 QML 插件的章节
- [Defining QML Types from C++](<./Defining QML Types from C++.md>) - 有关将 C++ 类型注册到运行时的信息。
- [How to Create Qt Plugins](https://doc.qt.io/qt-6/plugins-howto.html) - 有关 Qt 插件的信息





<!-- 完成标志, 看不到, 请忽略! -->