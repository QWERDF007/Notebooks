# [Using C++ Models with Qt Quick Views](https://doc.qt.io/qt-6/qtquick-modelviewsdata-cppmodels.html)

## Data Provided In A Custom C++ Model

可以在 C++  中定义模型，然后使其在 QML 中可用。这对于将现有的 C++ 数据模型或其他复杂数据集暴露给 QML 非常有用。

一个 C++ 模型类可以定义为 `QStringList`、`QVariantList`、`QObjectList` 或 `QAbstractItemModel`。前三个对于公开较简单的数据集非常有用，而 `QAbstractItemModel` 则为更复杂的模型提供了更灵活的解决方案。

这里有一个[视频教程](https://youtu.be/9BcAYDlpuT8)，全面介绍了如何将 C++ 模型暴露给QML。

### QStringList-based Model

一个模型可以是一个简单的 `QStringList`，它通过 `modelData` 角色提供列表的内容。

下面是一个 `ListView` 示例，它使用委托 (delegate) 通过 `modelData` 角色引用模型项的值:

```qml
ListView {
    width: 100
    height: 100
    required model

    delegate: Rectangle {
        required property string modelData
        height: 25
        width: 100
        Text { text: parent.modelData }
    }
}
```

一个 Qt 应用程序可用加载此 QML 文档并将 `myModel` 的值设置为一个 `QStringList`：

```c++
QStringList dataList = {
    "Item 1",
    "Item 2",
    "Item 3",
    "Item 4"
};

QQuickView view;
view.setInitialProperties({{ "model", QVariant::fromValue(dataList) }});
```

这个例子的完整源代码可在 Qt 安装目录下的 [examples/quick/models/stringlistmodel](https://doc.qt.io/qt-6/qtquick-models-stringlistmodel-example.html) 中找到。

> 注意：视图无法知道 `QStringList` 的内容已更改。如果 `QStringList` 更改，则需要通过再次设置视图的模型属性来重置模型。

### QVariantList-based Model

模型可以是单个 `QVariantList`，它通过 `modelData` 角色提供列表的内容。

它 API 的工作方式与 `QStringList` 相同，如前一节所示。

> 注意：视图无法知道 `QVariantList` 的内容已更改。如果 `QVariantList` 更改，则需要重置模型。

### QObjectList-based Model

一个 `QObject*` 值的列表也可以用作模型。 一个 `QList<QObject*>` 会将列表中对象的属性作为角色暴露出来。

下面的应用程序创建了一个 `DataObject` 类，其中包含 Q_PROPERTY 值，在通过 `QList<DataObject*>` 暴露给QML时，这些值可以作为命名角色被访问:

```c++
class DataObject : public QObject
{
    Q_OBJECT

    Q_PROPERTY(QString name READ name WRITE setName NOTIFY nameChanged)
    Q_PROPERTY(QString color READ color WRITE setColor NOTIFY colorChanged)
    ...
};

int main(int argc, char ** argv)
{
    QGuiApplication app(argc, argv);

    const QStringList colorList = {"red",
                                   "green",
                                   "blue",
                                   "yellow"};

    const QStringList moduleList = {"Core", "GUI", "Multimedia", "Multimedia Widgets", "Network",
                                    "QML", "Quick", "Quick Controls", "Quick Dialogs",
                                    "Quick Layouts", "Quick Test", "SQL", "Widgets", "3D",
                                    "Android Extras", "Bluetooth", "Concurrent", "D-Bus",
                                    "Gamepad", "Graphical Effects", "Help", "Image Formats",
                                    "Location", "Mac Extras", "NFC", "OpenGL", "Platform Headers",
                                    "Positioning", "Print Support", "Purchasing", "Quick Extras",
                                    "Quick Timeline", "Quick Widgets", "Remote Objects", "Script",
                                    "SCXML", "Script Tools", "Sensors", "Serial Bus",
                                    "Serial Port", "Speech", "SVG", "UI Tools", "WebEngine",
                                    "WebSockets", "WebView", "Windows Extras", "XML",
                                    "XML Patterns", "Charts", "Network Authorization",
                                    "Virtual Keyboard", "Quick 3D", "Quick WebGL"};

    QList<QObject *> dataList;
    for (const QString &module : moduleList)
        dataList.append(new DataObject("Qt " + module, colorList.at(rand() % colorList.length())));

    QQuickView view;
    view.setResizeMode(QQuickView::SizeRootObjectToView);
    view.setInitialProperties({{ "model", QVariant::fromValue(dataList) }});
    ...
```

`QObject*` 通过 `modelData` 属性可用。为方便使用，对象的属性也能直接在委托的上下文中可用。这里，`view.qml` 在 `ListView` 的委托中引用了 `DataModel` 的属性:

```qml
ListView {
    id: listview
    width: 200; height: 320
    required model
    ScrollBar.vertical: ScrollBar { }

    delegate: Rectangle {
        width: listview.width; height: 25

        required color
        required property string name

        Text { text: parent.name }
    }
}
```

注意到 `color` 属性的使用。您可以通过在派生类型中将它们声明为 `required` 来要求现有属性。

这个示例的完整源代码在 Qt 安装目录的 [examples/quick/models/objectlistmodel](https://doc.qt.io/qt-6/qtquick-models-objectlistmodel-example.html) 中。

> 注意：视图无法知道 `QList` 的内容已更改。如果 `QList` 更改，则需要通过再次设置模型属性来重置模型。

### QAbstractItemModel Subclass

可以通过子类化 `QAbstractItemModel` 来定义模型。如果您有一个更复杂的模型，无法通过其他方法支持，则这是最佳方法。`QAbstractItemModel` 还可以在模型数据更改时自动通知 QML 视图。

可以通过重实现 `QAbstractItemModel::roleNames()` 将 `QAbstractItemModel` 子类的角色暴露给QML。

下面是一个应用程序，有一个名为 `AnimalModel` 的 `QAbstractListModel` 子类，它暴露了 `type` 和 `size` 角色。它重新实现了 `QAbstractItemModel::roleNames()` 来暴露角色名，以便 QML 可以访问它们：

```c++
class Animal
{
public:
    Animal(const QString &type, const QString &size);
    ...
};

class AnimalModel : public QAbstractListModel
{
    Q_OBJECT
public:
    enum AnimalRoles {
        TypeRole = Qt::UserRole + 1,
        SizeRole
    };

    AnimalModel(QObject *parent = nullptr);
    ...
};

QHash<int, QByteArray> AnimalModel::roleNames() const {
    QHash<int, QByteArray> roles;
    roles[TypeRole] = "type";
    roles[SizeRole] = "size";
    return roles;
}

int main(int argc, char ** argv)
{
    QGuiApplication app(argc, argv);

    AnimalModel model;
    model.addAnimal(Animal("Wolf", "Medium"));
    model.addAnimal(Animal("Polar bear", "Large"));
    model.addAnimal(Animal("Quoll", "Small"));

    QQuickView view;
    view.setResizeMode(QQuickView::SizeRootObjectToView);
    view.setInitialProperties({{"model", QVariant::fromValue(&model)}});
    ...
```

这个模型由一个 `ListView` 委托显示，它访问 `type` 和 `size` 角色：

```qml
ListView {
    width: 200; height: 250

    required model

    delegate: Text {
        required property string type
        required property string size

        text: "Animal: " + type + ", " + size
    }
}
```

当模型更改时，QML 视图会自动更新。请记住，模型必须遵循模型更改的标准规则，并在模型更改时使用 `QAbstractItemModel::dataChanged()`、`QAbstractItemModel::beginInsertRows()` 等通知视图。更多信息请参阅 [模型子类化参考](../Qt Widgets/Model View Programming.md#Model Subclassing Reference)。

这个例子的完整源代码可以在 Qt 安装目录下的 examples/quick/models/abstractitemmodel 中找到。

`QAbstractItemModel` 呈现了一个表的层次结构，但是 QML 当前提供的视图只能显示列表数据。为了显示分层模型的子列表，请使用 `DelegateModel` QML 类型，该类型提供以下属性和函数，可与 `QAbstractItemModel` 类型的列表模型一起使用：

- `hasModelChildren` 角色属性用于确定节点是否具有子节点。
- `DelegateModel::rootIndex` 允许指定根节点
- `DelegateModel::modelIndex()` 返回一个可以赋值给 `DelegateModel::rootIndex` 的 `QModelIndex `
- `DelegateModel::parentModelIndex()` 返回一个可以赋值给 `DelegateModel::rootIndex` 的`QModelIndex`























