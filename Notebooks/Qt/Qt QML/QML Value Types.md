# [QML Value Types](https://doc.qt.io/qt-6/qtqml-typesystem-valuetypes.html)

QML 支持内置和自定义值类型。

值类型是指按值传递而不是按引用传递的类型，例如整数 (`int`) 或字符串 (`string`)。这与 [QML 对象类型](<./The QML Type System.md#QML Object Types>)形成对比。对象类型按引用传递。如果将对象类型的实例分配给两个不同的属性，则这两个属性都将携带相同的值。修改对象会反映在两个属性中。如果将值类型的实例分配给两个不同的属性，则这些属性将携带单独的值。如果您修改其中一个，另一个保持不变。与对象类型不同，值类型不能用于声明 QML 对象：例如，不可能声明一个 int {} 对象或一个 size {} 对象。

值类型可用于引用：

- 单个值（例如 `int` 引用单个数字） 
- 包含属性和方法的值（例如 `size` 引用具有 `width` 和 `height` 属性的值） 
- 通用类型 `var`。它可以容纳任何其他类型的的值，但它本身是值类型。 

当变量或属性持有值类型并将其分配给另一个变量或属性时，就会复制该值。

## Available Value Types

引擎默认支持某些值类型，无需使用 import 语句，而其他类型则需要客户端导入提供它们的模块。下面列出的所有值类型都可用作 QML 文档中的属性类型，但以下情况除外：

- `void`，表示没有值 
- `list` 必须与对象或值类型结合用作元素 
- `enumeration` 不能直接使用，因为枚举必须由注册的 QML 对象类型定义

QML 语言提供的内置值类型

下面列出了 QML 语言本身支持的内置值类型：

| 类型                                                       | 描述                                                    |
| ---------------------------------------------------------- | ------------------------------------------------------- |
| [bool](https://doc.qt.io/qt-6/qml-bool.html)               | Binary true/false value                                 |
| [date](https://doc.qt.io/qt-6/qml-date.html)               | Date value                                              |
| [double](https://doc.qt.io/qt-6/qml-double.html)           | Number with a decimal point, stored in double precision |
| [enumeration](https://doc.qt.io/qt-6/qml-enumeration.html) | Named enumeration value                                 |
| [int](https://doc.qt.io/qt-6/qml-int.html)                 | Whole number, e.g. 0, 10, or -20                        |
| [list](https://doc.qt.io/qt-6/qml-list.html)               | List of QML objects                                     |
| [real](https://doc.qt.io/qt-6/qml-real.html)               | Number with a decimal point                             |
| [string](https://doc.qt.io/qt-6/qml-string.html)           | A free form text string                                 |
| [url](https://doc.qt.io/qt-6/qml-url.html)                 | Resource locator                                        |
| [var](https://doc.qt.io/qt-6/qml-var.html)                 | Generic property type                                   |
| [variant](https://doc.qt.io/qt-6/qml-variant.html)         | Generic property type                                   |
| [void](https://doc.qt.io/qt-6/qml-void.html)               | Empty value type                                        |

### Value Types Provided By QML Modules

QML 模块可以扩展 QML 语言，提供更多数值类型。

QtQml 模块提供的数值类型：

| 类型                                           | 描述                                         |
| ---------------------------------------------- | -------------------------------------------- |
| [point](https://doc.qt.io/qt-6/qml-point.html) | Value with x and y attributes                |
| [rect](https://doc.qt.io/qt-6/qml-rect.html)   | Value with x, y, width and height attributes |
| [size](https://doc.qt.io/qt-6/qml-size.html)   | Value with width and height attributes       |

QtQuick 模块提供的数值类型：

| 类型                                                     | 描述                                                         |
| -------------------------------------------------------- | ------------------------------------------------------------ |
| [color](https://doc.qt.io/qt-6/qml-color.html)           | ARGB color value                                             |
| [font](https://doc.qt.io/qt-6/qml-font.html)             | Font value with the properties of `QFont`. The font type refers to a font value with the properties of `QFont` |
| [matrix4x4](https://doc.qt.io/qt-6/qml-matrix4x4.html)   | A matrix4x4 type is a 4-row and 4-column matrix              |
| [quaternion](https://doc.qt.io/qt-6/qml-quaternion.html) | A quaternion type has scalar, x, y, and z attributes         |
| [vector2d](https://doc.qt.io/qt-6/qml-vector2d.html)     | A vector2d type has x and y attributes                       |
| [vector3d](https://doc.qt.io/qt-6/qml-vector3d.html)     | Value with x, y, and z attributes                            |
| [vector4d](https://doc.qt.io/qt-6/qml-vector4d.html)     | A vector4d type has x, y, z and w attributes                 |

Qt 全局对象提供了一些用于操作数值类型值的实用函数。

您可以按照“用 C++ 定义 QML 类型”中的描述定义自己的数值类型。为了使用特定 QML 模块提供的类型，客户端必须在其 QML 文档中导入该模块。

## Property Change Behavior for Value Types

某些数值类型具有属性，例如 `font` 类型具有 `pixelSize`、`family` 和 `bold` 属性。与[对象类型](<./The QML Type System.md#QML Object Types>)的属性不同，数值类型的属性不会提供自己的属性更改信号。只能为数值类型属性本身创建属性更改信号处理程序：

```javascript
Text {
    // invalid!
    onFont.pixelSizeChanged: doSomething()

    // also invalid!
    font {
        onPixelSizeChanged: doSomething()
    }

    // but this is ok
    onFontChanged: doSomething()
}
```

但是请注意，只要数值类型的任何属性发生更改，以及属性本身更改时，都会发出数值类型的属性更改信号。例如以下代码：

```javascript
Text {
    onFontChanged: console.log("font changed")

    Text { id: otherText }

    focus: true

    // changing any of the font attributes, or reassigning the property
    // to a different font value, will invoke the onFontChanged handler
    Keys.onDigit1Pressed: font.pixelSize += 1
    Keys.onDigit2Pressed: font.b = !font.b
    Keys.onDigit3Pressed: font = otherText.font
}
```

相比之下，[对象类型](<./The QML Type System.md#QML Object Types>)的属性会发出自己的属性更改信号，并且只有在对象类型属性重新分配给不同的对象值时，才会调用对象类型属性的属性更改信号处理程序。

另请参阅[QML 类型系统](<./The QML Type System.md>)。



<!-- 完成标志, 看不到, 请忽略! -->