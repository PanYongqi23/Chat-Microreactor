# ChatReactor 项目部署指南
欢迎来到 ChatReactor 项目！以下是一份详细的部署指南，帮助您顺利地安装和运行本项目。
## 环境准备
### 1. 创建 Conda 环境
首先，您需要创建一个名为“ChatReactor”的 Conda 环境。打开命令行工具，输入以下命令：
```bash
conda create -n ChatReactor python=3.10
```
### 2. 激活环境
创建完成后，激活“ChatReactor”环境：
```bash
conda activate ChatReactor
```
### 3. 安装依赖
接下来，安装项目所需的依赖。本项目提供了一个 `environment.yml` 文件，您可以使用以下命令进行安装：
```bash
pip install -r requirements.txt
```
## 配置
### 4. 配置 OpenAI API
在项目目录下，找到 `data/accounts.txt` 文件。使用文本编辑器打开它，写入您的 OpenAI API 密钥，例如：
```
sk-wewqeqrsda1232dsadsdsads
```
请将上述示例替换为您的实际 API 密钥。
## 运行项目
### 5. 运行主程序
在完成上述步骤后，您可以运行项目的主程序。进入项目目录的 `src` 子目录，执行以下命令：
```bash
streamlit run .\main.py
```
运行成功后，您的浏览器将自动打开并显示 ChatReactor 的界面。
## 常见问题
- 如果遇到依赖安装问题，请确保您的 Conda 环境已经正确配置，并且网络连接正常。
- 如果运行主程序时出现错误，请检查 `data/accounts.txt` 文件中的 API 密钥是否正确。
感谢您使用 ChatReactor 项目，如有任何疑问或建议，请随时联系我们。祝您使用愉快！