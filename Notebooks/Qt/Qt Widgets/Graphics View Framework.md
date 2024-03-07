# [Graphics View Framework](https://doc.qt.io/qt-6/graphicsview.html)

Graphics View 提供了一种管理大量定制的 2D 图形 items 并与之交互的界面，以及一个用于可视化 items 的控件，支持缩放和旋转。

该框架包含一个事件传播体系结构，允许对场景中的 items 进行精准的双精度的交互。Items 可以处理按键事件，鼠标按下、移动、释放和双击事件，还可以跟踪鼠标的移动。

Graphics view 使用 BSP (二叉空间分割) 树来提供非常快速的 item 查找，因此，它可以实时可视化大型场景，即使有数百万个items。

Graphics View 是在 Qt 4.2 中引入的，取代了它的前身，`QCanvas`。

Graphics View 框架主要包含三个主要的类 `QGraphicsScene` (场景)、`QGraphicsView` (视图)、`QGraphicsItem` (图形项)。`QGraphicsScene` 本身不可见，是一个存储 items 的容器，必须通过与之关联的视图来可视化和与外界交互，提供 items 的操作接口、传递事件、管理 items 的状态和无变换的渲染功能 (如打印)；`QGraphicsView` 提供一个可视的窗口，用于显示场景中的 items，一个场景可以有多个视口；`QGraphicsItem` 是场景中的图形项的基类，Qt 提供了常用图形项的标注类。

## The Graphics View Architecture

Graphics View 为模型—视图编程提供了一种基于 item 的方法，很像 InterView 的便捷类 QTableView、QTreeView 和 QListView。多个视图可以观察单个场景，而场景包含不同几何形状的 items。

### The Scene

QGraphicsScene 提供了图形视图场景，功能如下：

- 提供管理大量items的接口
- 传播事件给场景中的每个items
- 管理items的状态，例如选中和焦点处理
- 提供无变换的渲染功能；主要用于打印

QGraphicsScene 充当 QGraphicsItem 对象的容器。通过调用 `QGraphicsScene::addItem()` 将 items 添加到场景中，然后调用多个 item 查找函数之一来进行检索。`QGraphics::Items()` 和它的重载返回一个点、矩形、多边形或者矢量路径包含或者相交的所有items。`QGraphicsScene::ItemAt()` 返回特定点的最顶层的 item。所有的 item 查找函数按递减的顺序返回 items (即，第一个返回的 item 是最上层的，而最后一个是最底层的)。

```c++
QGraphicsScene scene;
QGraphicsRectItem *rect = scene.addRect(QRectF(0, 0, 100, 100));

QGraphicsItem *item = scene.itemAt(50, 50, QTransform());
```

`QGraphicsScene` 的事件传播体系将场景事件传递给 items，并管理 items 之间的事件传播。如果场景在特定位置接收到鼠标按下事件，场景会将事件传递给位于该位置的 item。

`QGraphicsScene` 还管理 item 的某些状态，例如 item 的选中和焦点。你可以通过调用 `QGraphicsScene::setSelectionArea()` 并传递任意形状来选中场景中的 items。此功能也被用作  `QGraphicsView` 中橡皮筋选中的基础。调用 `QGraphicsScene::selectedItems()` 获取当前选中的 items 的列表。`QGraphicsScene` 处理的另外一个状态是一个 item 是否具有键盘输入焦点。你可以通过调用 `QGraphicsScene::setFocusItem()` 或者 `QGraphicsItem::setFocus()` 来设置焦点在某个 item 上，或者通过调用 `QGraphicsScene::focusItem()` 获取当前聚焦的 item。

最后，`QGraphicsScene` 允许你通过 `QGraphicsScene::render()` 函数将部分场景渲染到绘制设备中。你可以在文档后续部分的 Printing 章节中阅读更多相关信息。

### The View

QGraphicsView 提供视图控件，用于可视化场景中的内容。你可以将多个视图关联同一个场景，来为同一数据集提供多个视口。视图控件是一个滚动区域，并提供用于大场景中导航的滚动条。要启用 OpenGL，你可以通过调用 `QGraphicsView::setViewPort()` 将 `QOpenGLWidget` 作为视口。

```c++
QGraphicsScene scene;
myPopulateScene(&scene);
QGraphicsView view(&scene);
view.show()
```

视图接收来自键盘和鼠标的输入事件，在将事件发送到可视化场景之前，将这些事件转换为场景事件 (在适当的地方将所使用的坐标转换为场景坐标)。

使用它的变换矩阵，`QGraphicsView::transform()`，视图可以变换场景的坐标系。这允许高级导航功能，例如缩放和旋转。为了方便起见，`QGraphicsView` 还提供了视图和场景坐标之间的变换函数：`QGraphicsView::mapToScene()` 和 `QGraphicsView::mapFromScene()`。

<img src="./assets/graphicsview-view.png">

### The Item

`QGraphicsItem` 是场景中图形项的基类。Graphics View 为典型的形状提供了多个标准项，例如矩形 `QGraphicsRectItem`、椭圆 `QGraphicsEllipseItem` 和文本项 `QGraphicsTextItem`，最强大的`QGraphicsItem` 特性在你编写自定义的 items 可用。此外，`QGraphicsItem` 支持一下功能：

- 鼠标按下、移动、释放和双击事件，以及鼠标悬停事件、滚轮事件和上下文菜单事件
- 键盘输入焦点和按键事件
- 拖放
- 通过父子关系和 `QGraphicsItemGroup` 进行分组
- 碰撞检测

Items 存在于局部坐标系中，并且像 `QGraphicsView` 一样，它还提供了许多用于 item 和场景之间，以及 item 和 item 之间映射坐标的函数。此外，与 `QGraphicsView` 一样，它可以使用一个矩阵来变换它的坐标系：`QGraphicsItem::transform()`。这对于旋转和缩放单个 item 很有用。

Items 可以包含其他 items (子项)。所有的子 Item 继承父 Item 的变换。然而，无论一个 item 累积的变换如何，它的所有函数 (例如 `QGraphicsItem::contains()`、`QGraphicsItem::boundingRect()`、`QGraphicsItem::collidesWith()`) 仍在局部坐标系中执行。

`QGraphicsItem` 通过 `QGraphicsItem::shape()` 和 `QGraphicsItem::collidesWith()` 函数进行碰撞检测，这两个函数都是虚函数。通过从 `QGraphicsItem::shape()` 返回你的 item 的形状作为局部坐标 `QPainterPath`，`QGraphicsItem` 将为你处理所有的碰撞检测。但是，如果你想提供自己的碰撞检测，你可以重新实现 `QGraphicsItem::collidesWith()`。

<img src="./assets/graphicsview-items.png">

## Classes in the Graphics View Framework

| class                        | description                                          |
| ---------------------------- | ---------------------------------------------------- |
| `QAbstractGraphicsShapeItem` | 所有路径 items 的通用基类                            |
| `QGraphicsAnchor`            | 表示 `QGraphicsAnchorLayout` 中两个 items 之间的锚点 |
| `QGrphicsAnchorLayout`       | 将锚点控件放进 Graphics View 的布局控件              |
| `QGraphicsItem`              |                                                      |
|                              |                                                      |

| class                                                        | description                                                 |
| ------------------------------------------------------------ | ----------------------------------------------------------- |
| [QAbstractGraphicsShapeItem](https://doc.qt.io/qt-6/qabstractgraphicsshapeitem.html) | 所有路径 items 的通用基类                                   |
| [QGraphicsAnchor](https://doc.qt.io/qt-6/qgraphicsanchor.html) | 表示 `QGraphicsAnchorLayout` 中两个 items 之间的锚点        |
| [QGraphicsAnchorLayout](https://doc.qt.io/qt-6/qgraphicsanchorlayout.html) | 将锚点控件放进 Graphics View 的布局控件                     |
| [QGraphicsEffect](https://doc.qt.io/qt-6/qgraphicseffect.html) | 所有 graphics effects 的基类                                |
| [QGraphicsEllipseItem](https://doc.qt.io/qt-6/qgraphicsellipseitem.html) | 可以加入到 `QGraphicsScene` 的椭圆 item                     |
| [QGraphicsGridLayout](https://doc.qt.io/qt-6/qgraphicsgridlayout.html) | Graphics View 中的网格布局控件                              |
| [QGraphicsItem](https://doc.qt.io/qt-6/qgraphicsitem.html)   | `QGraphicsScene` 中所有图形项的基类                         |
| [QGraphicsItemGroup](https://doc.qt.io/qt-6/qgraphicsitemgroup.html) | 将一组 items 视为单个 item 的容器                           |
| [QGraphicsLayout](https://doc.qt.io/qt-6/qgraphicslayout.html) | Graphics View 中所有布局控件的基类                          |
| [QGraphicsLayoutItem](https://doc.qt.io/qt-6/qgraphicslayoutitem.html) | 可继承以允许管理自定义 items 的布局                         |
| [QGraphicsLineItem](https://doc.qt.io/qt-6/qgraphicslineitem.html) | 可以加入到 `QGraphicsScene` 的线 item                       |
| [QGraphicsLinearLayout](https://doc.qt.io/qt-6/qgraphicslinearlayout.html) | Graphics View 中的水平或垂直布局                            |
| [QGraphicsObject](https://doc.qt.io/qt-6/qgraphicsobject.html) | 所有需要信号、槽和属性的图形项的基类                        |
| [QGraphicsPathItem](https://doc.qt.io/qt-6/qgraphicspathitem.html) | 可以加入到 `QGraphicsScene` 的 Path item                    |
| [QGraphicsPixmapItem](https://doc.qt.io/qt-6/qgraphicspixmapitem.html) | 可以加入到 `QGraphicsScene` 的 Pixmap item                  |
| [QGraphicsPolygonItem](https://doc.qt.io/qt-6/qgraphicspolygonitem.html) | 可以加入到 `QGraphicsScene` 的多边形 item                   |
| [QGraphicsProxyWidget](https://doc.qt.io/qt-6/qgraphicsproxywidget.html) | Proxy layer for embedding a QWidget in a QGraphicsScene     |
| [QGraphicsRectItem](https://doc.qt.io/qt-6/qgraphicsrectitem.html) | 可以加入到 `QGraphicsScene` 的矩形 item                     |
| [QGraphicsScene](https://doc.qt.io/qt-6/qgraphicsscene.html) | 用于管理大量 2D 图形项的界面                                |
| [QGraphicsSceneContextMenuEvent](https://doc.qt.io/qt-6/qgraphicsscenecontextmenuevent.html) | 上下文菜单事件                                              |
| [QGraphicsSceneDragDropEvent](https://doc.qt.io/qt-6/qgraphicsscenedragdropevent.html) | 拖放事件                                                    |
| [QGraphicsSceneEvent](https://doc.qt.io/qt-6/qgraphicssceneevent.html) | 所有 graphics view 相关事件的基类                           |
| [QGraphicsSceneHelpEvent](https://doc.qt.io/qt-6/qgraphicsscenehelpevent.html) | 请求工具提示时的事件                                        |
| [QGraphicsSceneHoverEvent](https://doc.qt.io/qt-6/qgraphicsscenehoverevent.html) | graphics view 中的悬停事件                                  |
| [QGraphicsSceneMouseEvent](https://doc.qt.io/qt-6/qgraphicsscenemouseevent.html) | graphics view 中的鼠标事件                                  |
| [QGraphicsSceneMoveEvent](https://doc.qt.io/qt-6/qgraphicsscenemoveevent.html) | graphics view 中的控件的移动事件                            |
| [QGraphicsSceneResizeEvent](https://doc.qt.io/qt-6/qgraphicssceneresizeevent.html) | graphics view 中的控件的缩放事件                            |
| [QGraphicsSceneWheelEvent](https://doc.qt.io/qt-6/qgraphicsscenewheelevent.html) | graphics view 中的滚轮事件                                  |
| [QGraphicsSimpleTextItem](https://doc.qt.io/qt-6/qgraphicssimpletextitem.html) | 可以加入到 `QGraphicsScene` 的简单文本路径 item             |
| [QGraphicsSvgItem](https://doc.qt.io/qt-6/qgraphicssvgitem.html) | 可用于渲染 svg 文件内容的 item                              |
| [QGraphicsTextItem](https://doc.qt.io/qt-6/qgraphicstextitem.html) | 可以加入到 `QGraphicsScene` 的文本 item，以显示格式化的文本 |
| [QGraphicsTransform](https://doc.qt.io/qt-6/qgraphicstransform.html) | 用于在 `QGraphicsItem` 上构建高级变换的抽象基类             |
| [QGraphicsView](https://doc.qt.io/qt-6/qgraphicsview.html)   | 可视化 `QGraphicsScene` 内容的控件                          |
| [QGraphicsWidget](https://doc.qt.io/qt-6/qgraphicswidget.html) | `QGraphicsScene` 中所有控件的基类                           |
| [QStyleOptionGraphicsItem](https://doc.qt.io/qt-6/qstyleoptiongraphicsitem.html) | 用于描述绘制一个 `QGraphicsItem` 所需的参数                 |

## The Graphics View Coordinate System

Graphics View 基于笛卡尔坐标系；items 在场景中的位置和几何形状由两个数字的集合表示：x 坐标和 y 坐标。当使用一个未变换的视图观察场景时，屏幕上的一个像素表示场景中的一个单元。

**注意：不支持反转的 Y 轴坐标系，因为 Graphics View 使用 Qt 的坐标系**

Graphics View 中使用了三种有效的坐标系：Item 坐标系、场景坐标系和视图坐标系。为了简化你的实现，Graphics View 提供了便捷的函数在三个坐标系之间进行映射。

渲染时，Graphics View 的场景坐标对应 `QPainter` 的逻辑坐标，视图坐标等同于设备坐标。在[坐标系文档](https://doc.qt.io/qt-6/coordsys.html)中，你可以阅读逻辑坐标和设备坐标之间的关系。

<img src="./assets/graphicsview-parentchild.png">

### Item Coordinates

Item 存在于它们自身的局部坐标系中。它们的坐标通常以自身中心点 (0, 0) 作为原点，这也是所有变换的中心。Items 坐标系中的几何基元通常被称为 item points、item lines 或者 item rectangles。

创建自定义 item 时，你只需要担心 item 坐标；`QGraphicsScene` 和 `QGraphicsView` 将为你执行所有的转换。这使得实现自定义 item 变得非常容易。例如，如果你收到鼠标按下或拖动事件，则事件位置以 item 坐标给出。虚函数 `QGraphicsItem::contains()` 接受一个 item 坐标系中的点作为参数，如果某个点在你的 item 内部时返回 `true`，否则返回 `false`。同样，item 的包围矩形和形状也是 item 坐标系的。

Item 的位置是 item 的中心在其父 item 坐标系中的坐标；有时称为父坐标。从这个意义上说，场景被视为所有无父项的 items 的父项。顶级 items 的位置是场景坐标系的。

子 item 坐标是相对于父 item 坐标的。如果子 item 是未变换的，则子 item 坐标和父 item 坐标的差异与 items 在父坐标中的距离一致。例如：如果未变换的子 item 坐标精准地位于父 item 的中心点，则两个 items 的坐标系将相同。如果子 item 的位置是 (10,0)，则子 item 的 (0,10) 的点对应于父 item 的 (10,10) 的点。

因为 item 的位置和变换是相对于父 item 的，所以子 item 的坐标不受父 item 的变换的影响，尽管父 item 的变换隐式地变换了子 item。在上面的例子中，即使父 item 被旋转和缩放，子 item 的 (0,10) 点仍然对应父 item 的 (10,10) 点。然而，相对于场景，子 item 会跟随父 item 的变换和位置。如果父 item 被缩放 (2x, 2x)，则子 item 的位置将位于场景坐标 (20,0)，而它的 (10,0) 点将对应场景中的 (40,0) 点。

`QGraphicsItem::pos()` 是少数的例外之一，不管 item 和父 item 如何变换，`QGraphicsItem` 的函数都在 item 坐标中执行。例如，一个 item 的边界矩形 (即 `QGraphicsItem::boundingRect()`) 总是在 item 坐标中给出

### Scene Coordinates

场景表示其所有 items 的基本坐标系。场景坐标系描述了每个顶级 item 的位置，也构成了从视图传递到场景的所有场景事件的基础。除了局部的 item 位置和边界矩形，场景中的每个 item 还有一个场景的位置和边界矩形 (`QGraphics::scenePos()`，`QGraphicsItem::sceneBoundingRect()`)。场景位置描述了 item 在场景坐标中的位置，它的场景边界矩形构成了 `QGraphicsScene` 如何确定场景中哪些区域发生了变化的基础。场景中的变化通过 `QGraphicsScene::changed()` 信号进行通信，参数是场景矩形列表。

### View Coordinate

视图坐标是控件的坐标。视图坐标中每个单元对应一个像素。这个坐标系的特别之处在于它是相对于控件或者视口的，并且不受观察场景的影响。`QGraphicsView` 的视口的左上角始终为 (0,0)，右下角始终为 (视口宽，视口高)。所有的鼠标事件和拖放事件最初都是以视图坐标被接收的，你需要将这些坐标映射到场景中才能与 items 交互。

### Coordinate Mapping

通常在处理场景中的 items 时，将坐标和任意形状从场景坐标映射到 item 坐标，从 item 坐标映射到另一 item 坐标，或者从视图坐标映射到场景坐标是很有用的。例如，当你在 `QGraphicsView` 的视口中单击鼠标时，你可以通过调用 `QGraphicsView::mapToScene()` 后，跟着用 `QGraphicsScene::itemAt()` 来询问场景光标下的 item 是什么。如果你想知道 item 在视口中的位置，你可以在 item 上调用 `QGraphicsItem::mapToScene()`，然后在 view 上调用 `QGraphicsView::mapFromScene()`。最后，如果你想要查询视图椭圆内的 items，你可以传递一个 `QPainterPath` 给 `mapToScene()`，然后将映射后的路径传递给 `QGraphicsScene::items()`。

你可以通过调用 `QGraphicsItem::mapToScene()` 和 `QGraphicsItem::mapFromScene()` 将坐标和形状从 item 的场景映射和映射到 item 的场景。你还可以通过调用 `QGraphicsItem::mapToParent()` 和 `QGraphicsItem::mapFromParent()` 从父 item 映射和映射到父 item，或者通过调用 `QGraphicsItem::mapToItem()` 和 `QGraphicsItem::mapFromItem()` 在 items 之间映射。所有的映射函数都可以映射点、矩形、多边形和路径。

同样的映射函数 `QGraphicsView::mapFromScene()` 和 `QGraphicsView::mapToScene()`在视图中也可用，用于映射到场景和从场景映射。

## Key Features

### Zooming and rotating

`QGraphicsView` 通过 `QGraphicsView::setMatrix()` 支持与 `QPainter` 相同的仿射变换。通过对视图应用变换，你可以轻松地添加常见导航功能（例如缩放和旋转）的支持。

以下是如何在 `QGraphicsView` 的子类中实现缩放和旋转 slots 的示例：

```c++
class View : public QGraphicsView
{
Q_OBJECT
	...
public slots:
    void zoomIn() { scale(1.2, 1.2); }
    void zoomOut() { scale(1 / 1.2, 1 / 1.2); }
    void rotateLeft() { rotate(-10); }
    void rotateRight() { rotate(10); }
    ...
};
```

这些 slots 可以链接到启用 [autoRepeat](https://doc.qt.io/qt-6/qabstractbutton.html#autoRepeat-prop) 的 `QToolButtons`。

当你变换视图时，`QGraphicsView` 使视图的中心对齐。

另请参阅 [Elastic Notes](https://doc.qt.io/qt-6/qtwidgets-graphicsview-elasticnodes-example.html) 示例，了解如何实现基本缩放功能的代码。

### Ptinting

Graphics View 通过其渲染函数 `QGraphicsScene::render()` 和 `QGraphicsView::render()` 提供单行打印。这些函数提供相同的 API：通过传递一个 `QPainter` 任一渲染函数，你可以让场景或视图渲染全部或者部分内容到任意设备中。这个例子展示了如何使用 `QPainter` 将整个场景印刷成一个完整的页面。

```c++
QGraphicsScene scene;
QPainter printer;
scene.addRect(QRectF(0, 0, 100, 200), QPen(Qt::black), QBrush(Qt::green));

if (QPrintDialog(&printer).exec() == QDialog::Accepted) {
    QPainter painter(&printer);
    painter.setRenderHint(QPainter::Antialiasing);
    scene.render(&painter);
}
```

场景和视图渲染函数的区别在于，一个在场景坐标中运行，另一个在视图坐标中运行。`QGraphicsScene::render()` 通常首选用于印刷未转换的场景的整个片段，例如绘制几何数据，或者打印文本文档。另一方面，`QGraphicsView::render()` 适合截图；它的默认行为是使用提供的 painter 渲染视口的确切内容。

```c++
QGraphicsScene scene;
scene.addRect(QRectF(0, 0, 100, 200), QPen(Qt::black), QBrush(Qt::green));

QPixmap pixmap;
QPainter painter(&pixmap);
painter.setRenderHint(QPainter::Antialiasing);
scene.render(&painter);
painter.end();

pixmap.save("scene.png");
```

当源和目标区域大小不匹配时，源内容被拉伸以适合目标区域。通过将 `Qt::AspectRatioMode` 传递给你正在使用的渲染函数，你可以选择内容被拉伸时保持或者忽略场景的长宽比。

### Drag and Drop

因为 `QGraphicsView` 间接继承 `QWidget`，所以它已经提供了和 `QWidget` 一样的拖放功能。此外，为了方便起见，Graphics View 框架为场景和每个 item 也提供了拖放。当视图接收到 drag，它将拖放事件转换为 `QGraphicsSceneDragDropEvent`，然后将其转发给场景。场景接管这个事件的调度，并将它发送给鼠标光标下第一个接受 drops 的 item。

要开始一个 item 的拖动，创建一个 `QDrag` 对象，将指针传递给开始拖动的控件。Items 可以同时被多个视图观察，但只有一个视图可以开启拖动。在大多数情况下，拖动是由于按下或者移动鼠标而开始的，因此在 `mousePressEvent()` 和 `mouseMoveEvent()` 中，你可以从事件中获取原始控件指针。例如：

```c++
void CustomItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    QMimeData *data = new QMimeData;
    QDrag *drag = new QDrag(event->widgent());
    drag->setMimeData(data);
    drag->exec();
}
```

要拦截场景的拖放事件，你可以重新实现 `QGraphicsScene::dragEnterEvent()` 以及你的特定场景所需的任何事件处理程序。你可以在 `QGraphicsScene` 的每个事件处理程序的文档中阅读有关于在 Graphics View 中拖放的更多信息。

Items 可以通过调用 `QGraphicsItem::setAcceptDrops()` 启用拖放支持。要处理传入的拖动，重新实现 `QGraphicsItem::dragEnterEvent()`，`QGraphicsItem::dragMoveEvent()` ，`QGraphicsItem::dragLeaveEvent()` 和 `QGraphicsItem::dropEvent()`。

参阅 [Drag and Drop Robot](https://doc.qt.io/qt-6/qtwidgets-graphicsview-dragdroprobot-example.html) 示例，以演示 Graphics View 对拖放操作的支持。

### Cursors and Tooltips

与 `QWidget` 一样，`QGraphicsItem` 也支持光标（`QGraphicsItem::setCursor()`）和工具提示（`QGraphicsItem::setToolTip()`）。

当鼠标光标进入 item 的区域时（通过调用 `QGraphicsItem::contains()` 检测）`QGraphicsView` 会激活光标和工具提示。

你也可以通过调用 `QGraphicsView::setCursor()` 直接在视图上设置默认光标。

参阅 [Drag and Drop Robot](https://doc.qt.io/qt-6/qtwidgets-graphicsview-dragdroprobot-example.html) 示例，以了解实现工具提示和光标形状处理的代码。

### Animation

Graphics View 支持多个级别的动画。你可以使用动画框架轻松地组装动画。为此，你需要你的 item 继承 `QGraphicsObject` 并将 `QPropertyAnimation` 与它们相关联。`QpropertyAnimation` 允许为任何 `QObject` 属性设置动画。

另一种选择是创建一个继承 `QObject` 和 `QGraphicsItem` 的自定义 item。该 item 可以设置自己的定时器，并在 `QObject::timerEvent()` 中以增量步骤控制动画。

第三种选择主要用于与 Qt 3 种的 `QCanvas` 兼容，它通过调用 `QgraphicsScene::advance()` 推进场景，而 `QGraphicsScene::advance()` 又会调用 `QGraphicsItem::advance()`。

### OpenGL Rendering

要启用 OpenGL 渲染，你只需要通过调用 `QGraphicsView::setViewport()` 将一个新的 `QOpenGLWidget` 设置为 `QGraphicsView` 的视口。如果你希望 OpenGL 具有抗锯齿功能，你需要设置具有所需采样数的 `QSurfaceFormat` （参阅 `QGraphicsFormat::setSamples()`）。

例子：

```c++
QGraphicsView view(&scene);
QOpenGLWidget *gl = new QOpenGLWidget();
QSurfaceFormat format;
format.setSamples(4);
gl->setFormat(format);
view.setViewport(gl);
```

### Item Groups

通过使一个 item 成为另一个 item 的子项，你可以实现 item 分组的最本质的特性：items 将一起移动，并且所有转换都从父项传播到子项。

此外，`QGraphicsItemGroup` 是一个特殊的 item，通过一个有用的接口向组中添加 items 和从组中删除 items 合并子事件处理。将 item 添加到 `QGraphicsItemGroup` 将保持原始的位置和变换，而重新设置 item 的父 item 通常会导致子 item相对于其新的父 item 重新定位。为了方便，你可以通过调用 `QGraphicsScene::createItemGroup()` 通过场景创建 `QGraphicsItemGroup`。

### Widgets and Layouts

Qt 4.4 通过 `QGraphicsWidget` 引入了对几何和布局感知的 item 的支持。这个特殊的基类 item 类似于 `QWidget`，但与 `QWidget` 不同的是，它不继承自 `QPaintDevice`，而是继承自 `QGraphicsItem`。这允许你编写具有事件、信号和槽、大小提示和策略的完整控件，你还可以通过 `QGraphicsLinearLayout` 和 `QGraphicsGridLayout` 管理布局中控件的几何形状。

#### QGraphicsWidget

建立在 `QGraphicsItem` 的功能和 lean footprint 之上，`QGraphicsWidget` 提供了两者的优势：来自 `QWidget` 的额外功能，例如样式、字体、调色板、布局方向，以及它的几何形状，以及来自 `QGraphicsItem` 的分辨率的独立性和变换支持。因为 Graphics View 使用实数坐标而不是整数，`QGraphicsWidget` 的几何函数也在 `QRectF` 和 `QPointF` 上执行。这也适用于框架矩形、边距和间距。例如，使用 `QGraphicsWidget` 指定内容边距 (0.5, 0.5, 0.5, 0.5) 并不少见。你可以创建子控件和“顶级”窗口；在某些情况下，你现在可以将 QGraphics View 用于高级 MDI 应用程序。

支持某些 `QWidget` 的属性，包括窗口标志和属性，但不是全部。你应该参考 `QGraphicsWidget` 的[类文档](https://doc.qt.io/qt-6/qgraphicswidget.html)以全面了解支持和不支持的内容。例如，你可以将 `Qt::Window` 窗口标志传递给 `QGraphicsWidget` 的构造函数来创建装饰窗口，但 Graphics View 目前不支持 macOS 上场景的 `Qt::Sheet` 和 `Qt::Drawer` 标志。

#### QGraphicsLayout

`QGraphicsLayout` 是专门为 `QGraphicsWidget` 设计的第二代布局框架的一部分。它的 API 与 `QLayout` 的非常相似。你可以在 `QGraphicsLinearLayout` 和 `QGraphicsGridLayout` 中管理控件和子布局。你还可以子类化 `QGraphicsLayout` 来编写自己的布局，或者通过编写 `QGraphicsLayoutItem` 的适配器子类将自己的 `QGraphicsItem` items 添加到布局中。

### Embedded Widget Support

Graphics View 为将任何控件嵌入场景提供了无缝的支持。你可以嵌入简单的控件，例如 `QLineEdit` 和 `QPushButton`，复杂的控件，如 `QTabWidget`，甚至完整的主窗口。要将你的控件嵌入到场景中，只需调用 `QGraphicsScene::addWidget()`，或者创建 `QGraphicsProxyWidget` 的实例以手动地嵌入你的控件。

通过 `QGraphicsProxyWidget`，Graphics View 能够深度继承客户端控件特性，包括其光标、工具提示、鼠标、手写板和键盘事件、子控件、动画、弹出窗口（例如 `QComboBox` 或者 `QCompleter`），以及控件的输入焦点和激活。`QGraphicsProxyWidget` 甚至继承了嵌入的控件的 Tab 键顺序，因此你可以通过 Tab 键进入和退出嵌入的控件。你甚至可以将新的 `QGraphicsView` 嵌入你的场景以提供复杂的嵌套场景。

当变换一个嵌入的控件时，Graphics View 确保控件独立转换分辨率，允许字体和样式在放大时保持清晰。（注意，分辨率独立性的效果取决于样式。）

## Performance

### Floating Point Instructions

为了精确和快速地将变换和效果应用于 items，Graphics View 是在用户硬件能够为浮点指令提供合理性能的假设下构建的。

许多工作站和台式机都配备了合适的硬件来加速这种计算，但一些嵌入式设备可能只提供库来处理数学运算或在软件中模拟浮点指令。

因此，某些类型的效果在某些设备上可能比预期的要慢。可以通过在其他方面进行优化来补偿这种性能损失。例如，使用 `OpenGL` 渲染场景。但是，如果它们也依赖于浮点硬件的存在，那么任何此类的优化本身可能会导致性能下降。



<!-- 完成标志, 看不到, 请忽略! -->
