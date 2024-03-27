# [Data Type Conversion Between QML and C++]()

在 QML 和 C++ 之间交换数据值时，QML 引擎会将它们转换为适合在 QML 或 C++ 中使用的正确数据类型。这要求交换的数据属于引擎可识别的类型。

QML 引擎内置支持大量 Qt C++ 数据类型。此外，可以将自定义的 C++ 类型注册到 QML 类型系统中，以便引擎可以使用它们。

有关 C++ 和不同 QML 集成方法的更多信息，请参阅 [C++ 和 QML 集成概述](<./Overview - QML and C++ Integration.md>)。

本页讨论了 QML 引擎支持的数据类型以及它们如何在 QML 和 C++ 之间转换。

## Data Ownership

当数据从 C++ 传输到 QML 时，数据的所有权始终保留在 C++ 中。此规则的例外情况是通过显式的 C++ 方法调用返回 `QObject` 时：在这种情况下，QML 引擎假定拥有该对象，除非通过调用 `QQmlEngine::setObjectOwnership()` 并指定 `QQmlEngine::CppOwnership` 来显式地将对象的拥有权设置为保留在 C++ 中。

此外，QML 引擎会遵循 Qt C++ 对象的正常 `QObject`父对象所有权语义，并且永远不会删除具有父对象的 `QObject` 实例。

## Basic Qt Data Types

默认情况下，QML 识别以下 Qt 数据类型，这些类型在从 C++ 传递到 QML 以及反之亦然时会自动转换为对应的 QML 值类型：

| Qt Type                                                      | QML Value Type                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| bool                                                         | [bool](https://doc.qt.io/qt-6/qml-bool.html)                 |
| unsigned int, int                                            | [int](https://doc.qt.io/qt-6/qml-int.html)                   |
| double                                                       | [double](https://doc.qt.io/qt-6/qml-double.html)             |
| float, qreal                                                 | [real](https://doc.qt.io/qt-6/qml-real.html)                 |
| [QString](https://doc.qt.io/qt-6/qstring.html)               | [string](https://doc.qt.io/qt-6/qml-string.html)             |
| [QUrl](https://doc.qt.io/qt-6/qurl.html)                     | [url](https://doc.qt.io/qt-6/qml-url.html)                   |
| [QColor](https://doc.qt.io/qt-6/qcolor.html)                 | [color](https://doc.qt.io/qt-6/qml-color.html)               |
| [QFont](https://doc.qt.io/qt-6/qfont.html)                   | [font](https://doc.qt.io/qt-6/qml-font.html)                 |
| [QDateTime](https://doc.qt.io/qt-6/qdatetime.html)           | [date](https://doc.qt.io/qt-6/qml-date.html)                 |
| [QPoint](https://doc.qt.io/qt-6/qpoint.html), [QPointF](https://doc.qt.io/qt-6/qpointf.html) | [point](https://doc.qt.io/qt-6/qml-point.html)               |
| [QSize](https://doc.qt.io/qt-6/qsize.html), [QSizeF](https://doc.qt.io/qt-6/qsizef.html) | [size](https://doc.qt.io/qt-6/qml-size.html)                 |
| [QRect](https://doc.qt.io/qt-6/qrect.html), [QRectF](https://doc.qt.io/qt-6/qrectf.html) | [rect](https://doc.qt.io/qt-6/qml-rect.html)                 |
| [QMatrix4x4](https://doc.qt.io/qt-6/qmatrix4x4.html)         | [matrix4x4](https://doc.qt.io/qt-6/qml-matrix4x4.html)       |
| [QQuaternion](https://doc.qt.io/qt-6/qquaternion.html)       | [quaternion](https://doc.qt.io/qt-6/qml-quaternion.html)     |
| [QVector2D](https://doc.qt.io/qt-6/qvector2d.html), [QVector3D](https://doc.qt.io/qt-6/qvector3d.html), [QVector4D](https://doc.qt.io/qt-6/qvector4d.html) | [vector2d](https://doc.qt.io/qt-6/qml-vector2d.html), [vector3d](https://doc.qt.io/qt-6/qml-vector3d.html), [vector4d](https://doc.qt.io/qt-6/qml-vector4d.html) |
| Enums declared with [Q_ENUM](https://doc.qt.io/qt-6/qobject.html#Q_ENUM)() | [enumeration](https://doc.qt.io/qt-6/qml-enumeration.html)   |

> 注意：当包含了 Qt Quick 模块时，QML 才可以使用 Qt GUI 模块提供的类，例如 `QColor`、`QFont`、`QQuaternion` 和 `QMatrix4x4`。

为方便起见，许多这些类型可以在 QML 中通过字符串值指定，或者通过 `QtQml::Qt` 对象提供的相关方法指定。例如，`Image::sourceSize` 属性的类型为 `size`（会自动转换为 `QSize` 类型），可以通过格式为 "widthxheight" 的字符串值指定，也可以通过 `Qt.size()` 函数指定：

```
Item {
    Image { sourceSize: "100x200" }
    Image { sourceSize: Qt.size(100, 200) }
}
```

有关更多信息，请参阅 [QML 值类型](<./QML Value Types.md>) 下每个单独类型的文档。

## QObject-derived Types

任何继承自 `QObject` 的类都可以用作 QML 和 C++ 之间数据交换的类型，前提是该类已经注册到 QML 类型系统中。

引擎允许注册可实例化和不可实例化的类型。一旦类注册为 QML 类型，它就可以用作 QML 和 C++ 之间交换数据的类型。有关类型注册的更多详细信息，请参阅[使用 QML 类型系统注册 C++ 类型](<./Defining QML Types from C++.md#Registering C++ Types with the QML Type System>)。

## Conversion Between Qt and JavaScript Types

QML 引擎在 QML 和 C++ 之间传输数据时，内置了将许多 Qt 类型转换为相关 JavaScript 类型（反之亦然）的支持。这使得能够在 C++ 或 JavaScript 中使用这些类型并接收它们，而无需实现提供对数据值及其属性的访问的自定义类型。

请注意，QML 中的 JavaScript 环境会修改原生 JavaScript 对象原型，包括 `String`、`Date` 和 `Number`，以提供额外的功能。有关详细信息，请参阅 [JavaScript 主机环境](https://doc.qt.io/qt-6/qtqml-javascript-hostenvironment.html)。

### QVariantList and QVariantMap to JavaScript Array and Object

QML 引擎提供 `QVariantList` 和 JavaScript 数组之间以及 `QVariantMap` 和 JavaScript 对象之间的自动类型转换。

例如，下面用 QML 定义的函数期望两个参数，一个数组和一个对象，并使用标准的 JavaScript 语法访问数组和对象元素来打印它们的内容。下方的 C++ 代码调用此函数，传递了一个 `QVariantList` 和一个 `QVariantMap`，它们会自动分别转换为 JavaScript 数组和对象值：

QML：

```javascript
// MyItem.qml
Item {
    function readValues(anArray, anObject) {
        for (var i=0; i<anArray.length; i++)
            console.log("Array item:", anArray[i])

        for (var prop in anObject) {
            console.log("Object item:", prop, "=", anObject[prop])
        }
    }
}
```

C++

```c++
// C++
QQuickView view(QUrl::fromLocalFile("MyItem.qml"));

QVariantList list;
list << 10 << QColor(Qt::green) << "bottles";

QVariantMap map;
map.insert("language", "QML");
map.insert("released", QDate(2010, 9, 21));

QMetaObject::invokeMethod(view.rootObject(), "readValues",
        Q_ARG(QVariant, QVariant::fromValue(list)),
        Q_ARG(QVariant, QVariant::fromValue(map)));
```

这会产生类似这样的输出：

```bash
Array item: 10
Array item: #00ff00
Array item: bottles
Object item: language = QML
Object item: released = Tue Sep 21 2010 00:00:00 GMT+1000 (EST)
```

类似地，如果 C++ 类型将 `QVariantList` 或 `QVariantMap` 类型用于属性类型或方法参数，则可以在 QML 中将值创建为 JavaScript 数组或对象，并在将其传递给 C++ 时自动转换为 `QVariantList` 或 `QVariantMap`。

请注意，C++类型的 `QVariantList` 和 `QVariantMap` 属性存储为值，并且无法通过 QML 代码就地更改。您只能替换整个映射或列表，而不能修改其内容。以下代码如果属性 `l` 是 `QVariantList` 则不起作用：

```javascript
MyListExposingItem {
   l: [1, 2, 3]
   Component.onCompleted: l[0] = 10
}
```

下述代码生效：

```javascript
MyListExposingItem {
   l: [1, 2, 3]
   Component.onCompleted: l = [10, 2, 3]
}
```

### QDateTime to JavaScript Date

QML 引擎提供 `QDateTime` 值和 JavaScript `Date` 对象之间的自动类型转换。

例如，下面用 QML 定义的函数期望一个 JavaScript `Date` 对象，并且还返回一个包含当前日期和时间的新的 `Date` 对象。下方的 C++ 代码调用此函数，传递一个 `QDateTime` 值，该值在传递给 `readDate()` 函数时会由引擎自动转换为 `Date` 对象。反过来，`readDate()` 函数返回一个 `Date` 对象，该对象在 C++ 中接收时会自动转换为 `QDateTime` 值：

QML：

```javascript
// MyItem.qml
Item {
    function readDate(dt) {
        console.log("The given date is:", dt.toUTCString());
        return new Date();
    }
}
```

C++：

```c++
// C++
QQuickView view(QUrl::fromLocalFile("MyItem.qml"));

QDateTime dateTime = QDateTime::currentDateTime();
QDateTime retValue;

QMetaObject::invokeMethod(view.rootObject(), "readDate",
        Q_RETURN_ARG(QVariant, retValue),
        Q_ARG(QVariant, QVariant::fromValue(dateTime)));

qDebug() << "Value returned from readDate():" << retValue;
```

类似地，如果 C++ 类型将 `QDateTime` 用于属性类型或方法参数，则可以在 QML 中将值创建为 JavaScript `Date` 对象，并在将其传递给 C++ 时自动转换为 `QDateTime` 值。

> 注意：请注意月份编号的差异：JavaScript 将月份从 0 (一月) 编号到 11 (十二月)，与 Qt 将月份从 1 (一月) 编号到 12 (十二月) 的编号方式相差一个月。

> 注意：在 JavaScript 中使用字符串作为 `Date` 对象的值时，没有时间字段的字符串 (即简单的日期) 被解释为相关日期的 UTC 开始时间，这与 `new Date(y, m, d)` (使用当天本地时间开始时间) 不同。在 JavaScript 中构建 `Date` 对象的大多数其他方法都会产生本地时间，除非使用名称中包含 UTC 的方法。如果您的程序运行在落后于 UTC 的时区 (通常在主子午线以西)，则使用仅日期的字符串将导致 `Date` 对象的 `getDate()` 比字符串中的日期号少 1; 它通常会有一个较大的 `getHours()` 值。这些方法的 UTC 变体 `getUTCDate()` 和 `getUTCHours()` 将为您提供这类 `Date` 对象所期望的结果。请参阅下一节内容。

### QDate and JavaScript Date

QML 引擎通过将日期表示为其当天的 UTC 开始时间，在 `QDate` 和 JavaScript `Date` 类型之间进行自动转换。日期通过 `QDateTime` 映射回 `QDate`，选择其 `date()` 方法，使用日期的本地时间形式，除非 UTC 形式与下一天的开始时间一致，在这种情况下将使用 UTC 形式。

这种稍微古怪的安排是为了解决以下问题：JavaScript 从仅日期的字符串构建 `Date` 对象时使用当天的 UTC 开始时间，而 `new Date(y, m, d)` 则使用指定日期的本地时间开始时间（如上一节末尾的注意事项所述）。

因此，当 `QDate` 属性或参数暴露给 `QML` 时，读取其值时应该小心：`Date.getUTCFullYear()`、`Date.getUTCMonth()` 和 `Date.getUTCDate()` 方法比不带 UTC 的对应方法更可能提供用户期望的结果。

因此，通常更健壮的做法是使用 `QDateTime` 属性。这使得能够在 `QDateTime` 方面控制日期（和时间）是根据 UTC 还是本地时间指定的；只要 JavaScript 代码编写成遵循相同的标准，就应该可以避免出现问题。

### QTime and JavaScript Date

QML 引擎提供 `QTime` 值到 JavaScript `Date` 对象的自动类型转换。由于 `QTime` 值不包含日期部分，因此在转换过程中会创建一个日期部分。因此，您不应该依赖结果 `Date` 对象的日期部分。

在底层，从 JavaScript `Date` 对象转换为 `QTime` 的操作是通过转换为 `QDateTime` 对象（使用本地时间）并调用其 `time()` 方法来完成的。

### Sequence Type to JavaScript Array

QML 透明地支持某些 C++ 序列类型，使其表现得像 JavaScript `Array` 类型。

具体来说，QML 目前支持：

- `QList<int>`
- `QList<qreal>`
- `QList<bool>`
- `QList<QString>` 和 `QStringList`
- `QVector<QString>`
- `std::vector<QString>`
- `QList<QUrl>`
- `QVector<QUrl>`
- `std::vector<QUrl>`
- `QVector<int>`
- `QVector<qreal>`
- `QVector<bool>`
- `std::vector<int>`
- `std::vector<qreal>`
- `std::vector<bool>`

以及所有注册的包含用 `Q_DECLARE_METATYPE` 标记的类型的 `QList`、`QVector`、`QQueue`、`QStack`、`QSet`、`std::list`、`std::vector`。

这些序列类型直接通过底层的 C++ 序列实现。暴露此类序列给 QML 有两种方式：作为给定序列类型的 `Q_PROPERTY`；或者作为 `Q_INVOKABLE` 方法的返回值类型。这些实现方式之间存在一些重要差异，需要特别注意。

如果序列被暴露为 `Q_PROPERTY`，则通过索引访问序列中的任何值都会导致从 `QObject` 的属性读取序列数据，然后进行读取操作。类似地，修改序列中的任何值都会导致读取序列数据，然后进行修改，并将修改后的序列写回 `QObject` 的属性。

如果序列是从 `Q_INVOKABLE` 函数返回的，则访问和修改的代价要低得多，因为不会发生 `QObject` 属性的读取或写入；而是直接访问和修改 C++ 序列数据。

在 `Q_PROPERTY` 和 `Q_INVOKABLE` 返回值两种情况下，`std::vector` 的元素都会被复制。这种复制可能是一个昂贵的操作，因此应谨慎使用 `std::vector`。

您还可以通过使用 `QJSEngine::newArray()` 构造一个 `QJSValue` 来创建类似于列表的数据结构。这样的 JavaScript 数组在 QML 和 C++ 之间传递时不需要任何转换。有关如何从 C++ 操作 JavaScript 数组的详细信息，请参阅 [QJSValue#Working With Arrays](https://doc.qt.io/qt-6/qjsvalue.html#working-with-arrays)。

其他序列类型不被透明地支持，任何其他序列类型的实例将在 QML 和 C++ 之间作为不透明的 `QVariantList` 传递。

重要注意事项：由于实现中使用了 C++ 存储类型，因此此类序列数组类型和默认 JavaScript 数组类型之间存在一些细微的语义差异。特别是，从数组中删除一个元素将导致用默认构造的值替换该元素，而不是 `undefined` 值。类似地，将数组的 `length` 属性设置为大于其当前值的值将导致数组用默认构造的元素填充到指定长度，而不是 `undefined` 元素。最后，Qt 容器类支持有符号（而非无符号）整型索引；因此，尝试访问任何大于 `INT_MAX` 的索引都会失败。

每个序列类型的默认构造值如下：

| 类型                                | 默认构造值                                           |
| ----------------------------------- | ---------------------------------------------------- |
| `QList<int>`                        | integer value 0                                      |
| `QList<qreal>`                      | real value 0.0                                       |
| `QList<bool>`                       | boolean value `false`                                |
| `QList<QString>` 和 `QStringList  ` | empty [QString](https://doc.qt.io/qt-6/qstring.html) |
| `QVector<QString>`                  | empty [QString](https://doc.qt.io/qt-6/qstring.html) |
| `std::vector<QString>`              | empty [QString](https://doc.qt.io/qt-6/qstring.html) |
| `QList<QUrl>`                       | empty [QUrl](https://doc.qt.io/qt-6/qurl.html)       |
| `QVector<QUrl>`                     | empty [QUrl](https://doc.qt.io/qt-6/qurl.html)       |
| `std::vector<QUrl> `                | empty [QUrl](https://doc.qt.io/qt-6/qurl.html)       |
| `QVector<int>`                      | integer value 0                                      |
| `QVector<qreal> `                   | real value 0.0                                       |
| `QVector<bool>`                     | boolean value `false`                                |
| std::vector<int>                    | integer value 0                                      |
| `std::vector<qreal>`                | real value 0.0                                       |
| `std::vector<bool>`                 | boolean value `false`                                |

如果要从序列中删除元素，而不是简单地用默认构造的值替换它们，请不要使用索引删除运算符 (`delete sequence[i]`)，而是使用 `splice` 函数 (`sequence.splice(startIndex, deleteCount)`)。

### QByteArray to JavaScript ArrayBuffer

QML 引擎提供 `QByteArray` 值和 JavaScript `ArrayBuffer` 对象之间的自动类型转换。

### Value Types

Qt 中的一些值类型（例如 `QPoint`）在 JavaScript 中表示为具有与 C++ API 中相同的属性和函数的对象。自定义 C++ 值类型也可以使用相同的表示形式。为了使自定义值类型与 QML 引擎一起使用，类声明需要用 `Q_GADGET` 进行注释。希望在 JavaScript 表示中可见的属性需要用 `Q_PROPERTY` 声明。类似地，函数需要用 `Q_INVOKABLE` 标记。这与基于 `QObject` 的 C++ API 相同。例如，下面的 `Actor` 类被注释为 gadget，并具有属性：

```c++
class Actor
{
    Q_GADGET
    Q_PROPERTY(QString name READ name WRITE setName)
public:
    QString name() const { return m_name; }
    void setName(const QString &name) { m_name = name; }

private:
    QString m_name;
};

Q_DECLARE_METATYPE(Actor)
```

通常的模式是将 gadget 类用作属性的类型，或者将 gadget 作为信号参数发出。在这种情况下，gadget 实例在 C++ 和 QML 之间按值传递 (因为它是值类型)。如果 QML 代码更改了 gadget 属性的属性，则会重新创建整个 gadget 并将其传递回 C++ 属性设置器。在 Qt 5 中，gadget 类型不能通过在 QML 中直接声明来实例化。相比之下，可以声明一个 `QObject` 实例；`QObject` 实例总是通过指针从 C++ 传递到 QML。



## Enumeration Types

要将自定义枚举用作数据类型，其类必须注册，并且枚举也必须使用 `Q_ENUM()` 声明以将其注册到 Qt 的元对象系统。例如，下面的 `Message` 类具有一个 `Status` 枚举：

```c++
class Message : public QObject
{
    Q_OBJECT
    Q_PROPERTY(Status status READ status NOTIFY statusChanged)
public:
    enum Status {
        Ready,
        Loading,
        Error
    };
    Q_ENUM(Status)
    Status status() const;
signals:
    void statusChanged();
};
```

如果 `Message` 类已使用 QML 类型系统注册，则其 `Status` 枚举可以从 QML 中使用：

```javascript
Message {
     onStatusChanged: {
         if (status == Message.Ready)
             console.log("Message is loaded!")
     }
 }
```

有关在 QML 中将枚举用作 [flags](https://doc.qt.io/qt-6/qflags.html) 类型，请参阅 [Q_FLAG()](https://doc.qt.io/qt-6/qobject.html#Q_FLAG)。

> 注意：枚举值的名称必须以大写字母开头才能从 QML 访问。

```c++
...
enum class Status {
          Ready,
          Loading,
          Error
}
Q_ENUM(Status)
...
```

枚举类在 QML 中注册为作用域和非作用域属性。`Ready` 值将注册在 `Message.Status.Ready` 和 `Message.Ready`。

使用枚举类时，可以有多个枚举使用相同的标识符。非作用域注册将被最后一个注册的枚举覆盖。对于包含此类名称冲突的类，可以通过使用特殊的 `Q_CLASSINFO` 宏来禁用非作用域注册。使用名称 `RegisterEnumClassesUnscoped` 并将其值设置为 `false` 以防止作用域枚举合并到相同的命名空间。

```c++
class Message : public QObject
{
    Q_OBJECT
    Q_CLASSINFO("RegisterEnumClassesUnscoped", "false")
    Q_ENUM(ScopedEnum)
    Q_ENUM(OtherValue)

public:
    enum class ScopedEnum {
          Value1,
          Value2,
          OtherValue
    };
    enum class OtherValue {
          Value1,
          Value2
    };
};
```

来自相关类型的枚举通常在相关类型的范围内注册。例如， `Q_PROPERTY` 声明中使用的来自不同类型的任何枚举都会导致来自该类型的所有枚举都可以在 QML 中使用。这通常更像是一个缺陷而不是一个特性。为了防止这种情况发生，请使用特殊的 `Q_CLASSINFO` 宏注释您的类。使用名称 `RegisterEnumsFromRelatedTypes` 并将其值设置为 `false` 以防止来自相关类型的枚举在此类型中注册。

您应该显式注册任何想要在 QML 中使用的枚举的包围类型，使用 `QML_ELEMENT` 或 `QML_NAMED_ELEMENT`，而不是依赖于它们的枚举被注入到其他类型中。

```c++
class OtherType : public QObject
{
    Q_OBJECT
    QML_ELEMENT

public:
    enum SomeEnum { A, B, C };
    Q_ENUM(SomeEnum)

    enum AnotherEnum { D, E, F };
    Q_ENUM(AnotherEnum)
};

class Message : public QObject
{
    Q_OBJECT
    QML_ELEMENT

    // This would usually cause all enums from OtherType to be registered
    // as members of Message ...
    Q_PROPERTY(OtherType::SomeEnum someEnum READ someEnum CONSTANT)

    // ... but this way it doesn't.
    Q_CLASSINFO("RegisterEnumsFromRelatedTypes", "false")

public:
    OtherType::SomeEnum someEnum() const { return OtherType::B; }
};
```

重要区别在于 QML 中枚举的作用域。如果来自相关类的枚举自动注册，则作用域是其导入到的类型。例如，在上文中，如果没有额外的 `Q_CLASSINFO`，您可以使用 `Message.A`。如果持有枚举的 C++ 类型被显式注册，并且来自相关类型的枚举的注册被抑制，那么持有枚举的 C++ 类型的 QML 类型就是其所有枚举的作用域。您将在 QML 中使用 `OtherType.A` 而不是 `Message.A`。

请注意，您可以使用 `QML_FOREIGN` 注册无法修改的类型。您还可以使用 `QML_FOREIGN_NAMESPACE` 将 C++ 类型的枚举器注册到任何大写名称的 QML 名称空间中，即使相同的 C++ 类型也注册为 QML 值类型。

### Enumeration Types as Signal and Method Parameters

声明枚举类型和信号或方法都在同一个类中，或者枚举值是 [Qt 命名空间](https://doc.qt.io/qt-6/qml-qtqml-qt.html)中声明的枚举值之一，则可以使用枚举类型参数的 C++ 信号和方法。

此外，如果带枚举参数的 C++ 信号应该可以使用 `connect()` 函数连接到 QML 函数，则枚举类型必须使用 `qRegisterMetaType()` 注册。

对于 QML 信号，可以使用 `int` 类型将枚举值作为信号参数传递：

```javascript
Message {
    signal someOtherSignal(int statusValue)

    Component.onCompleted: {
        someOtherSignal(Message.Loading)
    }
}
```



<!-- 完成标志, 看不到, 请忽略! -->