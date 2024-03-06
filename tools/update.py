from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import importlib

FOLDERS = [
    'Papers',
    'Notebooks',
    'Code',
    'Flows',
]

EXCLUDE_LIST = [
    '.git',
    '.gitignore'
]

EMOJI = {
    'check': ':white_check_mark:',
    'cross': ':x:',
}

def get_file_mtime(file: Path) -> float:
    """获取文件修改时间

    Args:
        file (Path): 文件路径        
        
    Returns:
        float: 修改时间
    """
    if isinstance(file, str):
        file = Path(file)
    return file.stat().st_mtime

def format_mtime(mtime: float, format: str ='%Y-%m-%d') -> str:
    """格式化文件修改时间

    Args:
        mtime (float): 文件修改时间
        format (str, optional): 时间格式. 默认时间格式 '%Y-%m-%d'.

    Returns:
        str: 修改时间
    """
    modified_datetime = datetime.fromtimestamp(mtime)
    return modified_datetime.strftime(format)

def get_file_mtime_str(file: Path, format: str ='%Y-%m-%d') -> str:
    """获取文件修改时间, 并格式化输出

    Args:
        file (Path): 文件路径
        format (str, optional): 时间格式. 默认时间格式 '%Y-%m-%d'.

    Returns:
        str: 修改时间
    """
    return format_mtime(get_file_mtime(file), format)

def get_files(folder : Path, exclude_list: list = []) -> list:
    """递归遍历文件夹，获取所有文件，剔除传入的排除列表

    Args:
        folder (Path): 要递归的目录
        exclude_list (list, optional): 剔除列表. 默认 [].

    Returns:
        list: 递归目录下所有文件列表
    """
    files = []
    if isinstance(folder, str):
        folder = Path(folder)
    for file in folder.iterdir():
        if file.is_file() and file.name not in exclude_list:
            files.append(file)
        elif file.is_dir() and file.name not in exclude_list:
            files.extend(get_files(file, exclude_list))
    return files

def get_files_count(folder : Path, exclude_list: list = []) -> int:
    """获取目录下的文件数量，剔除排除列表

    Args:
        folder (Path): 目录
        exclude_list (list, optional): 排除列表. 默认 [].

    Returns:
        int: 文件数量
    """
    return len(get_files(folder, exclude_list))


class ReadmeGenerator(ABC):
    def __init__(self):
        """生成 README 内容
        """
        pass

    @abstractmethod
    def content(self):
        """返回 README 的内容
        """
        pass

class TitleGenerator(ReadmeGenerator):
    def __init__(self):
        super().__init__()

    def content(self):
        title = '# Notebooks\n\n'
        return title
    
class DescriptionGenerator(ReadmeGenerator):
    def __init__(self):
        super().__init__()

    def content(self):
        description = '一些笔记、流程图、调用示例、论文翻译、心得等等\n\n'
        table = '| Papers | Notebooks | Code | Flows |\n'
        table += '| --- | --- | --- | --- |\n'
        table += '| [:book:](#book-Papers) ' \
                 '| [:notebook:](#notebook-Notebooks) ' \
                 '| [:computer:](#computer-Code) ' \
                 '| [:traffic_light:](#traffic_light-Flows) ' \
                 '| \n\n'
        content = description + table
        return content

class FolderGenerator(ReadmeGenerator):
    def __init__(self, folder: str, icon: str = '',  desc: str = '', exclude_list: list = []):
        """遍历 [folder] 目录, 生成 [folder] 内的内容到 README, 剔除排除列表的文件

        Args:
            folder (str): 目录
            exclude_list (list, optional): 排除文件列表. 默认 [].
        """
        super().__init__()
        self.folder = folder
        self.icon = icon
        self.exclude_list = exclude_list
        self.readme = f'./{folder}/README.md'
        self.desc = desc

    def title(self) -> str:
        return f'## {self.icon} {self.folder}\n\n'

    def content(self, with_whats_new: bool = False):
        if with_whats_new:
            content = self.title() + self.table() + self.whats_new()
        else:
            content = self.title() + self.table()
        return content
    
    def table(self):
        return self.table_title() + self.table_content()

    def table_content(self):
        content = f'| {self.folder} | {self.desc} | {self.count()} | [README :link:](<{self.readme}>) |\n'
        return content

    def table_title(self):
        title = '| 目录 | 描述 | 数量 | 链接 |\n'
        title += '| --- | --- | --- | --- |\n'
        return title

    def count(self) -> int:
        """获取目录下的文件数量，剔除排除列表

        Args:
            folder (Path): 目录
            exclude_list (list, optional): 排除列表. Defaults to [].

        Returns:
            int: 文件数量
        """
        return len(get_files(self.folder, self.exclude_list))
    
    def whats_new(self) -> str:
        """获取目录下最新文件的内容"""
        files = get_files(self.folder, self.exclude_list)
        content = "### What's New ?\n\n"
        content += "<details><summary><em>[点击展开]</em></summary>\n"\
                   "<br>\n\n"
        if len(files) == 0:
            return content
        
        mtimes = [get_file_mtime(file) for file in files]
        sorted_files_mtimes = [(mtime,file) for mtime, file in sorted(zip(mtimes, files), reverse=True)]

        
        
        for mtime, file in sorted_files_mtimes:
            content += f'- {format_mtime(mtime)} [{file.name}](<{file.relative_to(self.folder).as_posix()}>)\n'

        content += "\n</details>\n\n"
        return content


if __name__ == '__main__':
    print(f"Working directory: {Path.cwd()}")
    print("Generating README.md...")

    with open('Papers/README.md', 'w', encoding='utf-8') as f:
        papers_generator = FolderGenerator('Papers', ":book:", '论文翻译, 主要是计算机视觉的相关论文', ['assets'])
        content = papers_generator.content()
        content += papers_generator.whats_new()
        f.write(content)
    
    with open('Notebooks/README.md', 'w', encoding='utf-8') as f:
        notebooks_generator = FolderGenerator('Notebooks', ":notebook:", '杂七杂八的笔记', ['assets'])
        content = notebooks_generator.content()
        content += notebooks_generator.whats_new()
        f.write(content)

    with open('Code/README.md', 'w', encoding='utf-8') as f:
        code_generator = FolderGenerator('Code', ":computer:", '一些修改过的代码以及一些算法实现', ['assets'])
        content = code_generator.content()
        content += code_generator.whats_new()
        f.write(content)

    with open('Flows/README.md', 'w', encoding='utf-8') as f:
        flows_generator = FolderGenerator('Flows', ":traffic_light:", '一些流程图', ['assets'])
        content = flows_generator.content()
        content += flows_generator.whats_new()
        f.write(content)
    
    
    content = TitleGenerator().content()
    content += DescriptionGenerator().content()
    content += papers_generator.content()
    content += notebooks_generator.content()
    content += code_generator.content()
    content += flows_generator.content()
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(content)

    print("Done!")

    

    