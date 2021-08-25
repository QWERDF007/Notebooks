[TOC]

# 目标检测数据格式

## COCO

[Data format](https://cocodataset.org/#format-data)

COCO 有几种标注类型：用于[目标检测](https://cocodataset.org/#detection-2020)、[关键点检测](https://cocodataset.org/#keypoints-2020)、[stuff 分割](https://cocodataset.org/#stuff-2019) (草、墙、天空等)、[全景分割](https://cocodataset.org/#panoptic-2020)、[密集姿势](https://cocodataset.org/#densepose-2020)和[图像描述](https://cocodataset.org/#captions-2015)。标注用 [JSON](https://www.json.org/json-en.html) 存储。[下载](https://cocodataset.org/#download)页面中描述的 [COCO API](https://github.com/cocodataset/cocoapi) 可用于访问和操作所有的标注。所有的标注共享以下相同的基础数据结构：

```python
{
	"info": info, 
	"images": [image], 
	"annotations": [annotation], 
	"licenses": [license],
}

info{
	"year": int, 
	"version": str, 
	"description": str, 
	"contributor": str, 
	"url": str, 
	"date_created": datetime,
}

image{
	"id": int, 
    "width": int, 
    "height": int, 
    "file_name": str, 
    "license": int, 
    "flickr_url": str, 
    "coco_url": str, 
    "date_captured": datetime,
}

license{
	"id": int, 
    "name": str, 
    "url": str,
}
```

[coco api example](./examples/coco_api_detection_example.ipynb)

下面描述了特定于各种标注类型的数据结构。

### 1. Object Detection

每个物体实例标注包含一系列字段，包括物体的类别 id 和分割掩码。分割格式取决于实例时代表单个对象 (iscrowd=0，这种情况下用多边形) 还是一组对象 (iscrowd=1，这种情况下用 RLE)。单个对象 (iscrowd=0) 可能需要多个多边形，例如被遮挡。群体标注 (iscrowd=1) 用于标注大量对象 (例如人群)。此外，为每个对象提供了一个封闭的边界框 (框的坐标从图像的左上角开始测量，并且是0索引的)。最后，标注的数据结构的类别字段存储类别 id 来映射到类别名称和超类的名称。另见 [detection](https://cocodataset.org/#detection-2020) 任务。

```python
annotation{
	"id": int, 
	"image_id": int, 
	"category_id": int, 
	"segmentation": RLE or [polygon], 
	"area": float, 
	"bbox": [x,y,width,height], 
	"iscrowd": 0 or 1,
}

categories[{
	"id": int, 
	"name": str, 
	"supercategory": str,
}]
```

## VOC

[Development Kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/index.html)

VOC 有五个主要任务：分类、检测、分割、行为分类和大规模识别。此外，有两个`taster` 任务，无框动作分类，人物布局。

有20个类别：人、鸟、猫、牛、狗、马、羊、飞机、自行车、船、巴士、汽车、摩托车、火车、瓶子、椅子、餐桌、盆栽、沙发、电视/显示器

VOC 为每个图像提供一个 xml 格式的标注文件。

```xml
<annotation>
	<folder>VOC2008</folder>
	<filename>2008_003483.jpg</filename>
	<source>
		<database>The VOC2008 Database</database>
		<annotation>PASCAL VOC2008</annotation>
		<image>flickr</image>
		<flickrid>n/a</flickrid>
	</source>
	<owner>
		<flickrid>n/a</flickrid>
		<name>n/a</name>
	</owner>
	<size>
		<width>500</width>
		<height>500</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>person</name>
		<pose>Frontal</pose>
		<truncated>0</truncated>
		<occluded>0</occluded>
		<bndbox>
			<xmin>304.1916</xmin>
			<ymin>118.2635</ymin>
			<xmax>482.0359</xmax>
			<ymax>500</ymax>
		</bndbox>
		<difficult>0</difficult>
		<part>
			<name>head</name>
			<bndbox>
				<xmin>391.1026</xmin>
				<ymin>120.3603</ymin>
				<xmax>433.1332</xmax>
				<ymax>171.6703</ymax>
			</bndbox>
		</part>
		<part>
			<name>hand</name>
			<bndbox>
				<xmin>303.7664</xmin>
				<ymin>194.5961</ymin>
				<xmax>332.1507</xmax>
				<ymax>224.6179</ymax>
			</bndbox>
		</part>
	</object>
</annotation>

```

`source` 说明图像来源，`size` 说明图像大小，`object`  说明标注物体，一副图像可有多个物体，`bndbox` 包含该物体的包围矩形的左上、右下坐标，`part` 指定人的特定部分，即头/手/脚。

## YOLO

yolo的数据格式为每幅图像一个txt标注文件，文件内每行为一个框的标注信息，其中坐标信息是关于图像大小归一化的，即 x，y 分别除以图像宽、高。

```txt
<object-class> <x> <y> <width> <height>
```

