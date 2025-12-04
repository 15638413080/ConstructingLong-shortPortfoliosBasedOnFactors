操作系统： Linux(Ubuntu 20.04)
python version: Python 3.8.5
安装软件包：requirements.txt

1. 第一次克隆仓库（Clone）
bash
# 克隆仓库到本地
git clone https://github.com/15638413080/ConstructingLong-shortPortfoliosBasedOnFactors.git

# 进入项目目录
cd ConstructingLong-shortPortfoliosBasedOnFactors

# 查看当前分支（应该是 main）
git branch -a
2. 拉取最新代码（Pull）
bash
# 从远程 main 分支拉取最新代码
git pull origin main

# 或者简写（如果已经设置 upstream）
git pull
3. 提交代码到仓库（Push）
bash
# 1. 查看当前状态
git status

# 2. 添加要提交的文件
git add .                   # 添加所有修改的文件
# 或者
git add 文件名              # 添加特定文件
git add *.py               # 添加所有.py文件

# 3. 提交到本地仓库
git commit -m "提交说明：添加MAX因子分析功能"

# 4. 推送到远程 main 分支
git push origin main

# 如果设置了上游分支，可以简写
git push




================================================================================
