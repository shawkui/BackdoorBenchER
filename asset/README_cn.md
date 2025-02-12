# 评估和重新审视后门净化中的辅助数据

[English](../README.md) | [简体中文](./README_cn.md)

[论文](https://arxiv.org/abs/2502.07231) | [引用](#citation)

---

## 📢 公告

**更新于 2025-02-12**：初始版本现已发布。支持对分类为已见（训练）、保留（分割）和OOD（变换）的辅助数据集上的后门净化进行评估。

---

## 📝 简介

欢迎来到“重新审视后门净化中的辅助数据”论文的官方代码库。本项目旨在通过使用多样的辅助数据集建立一个在实际条件下评估后门净化技术的框架，打破理想化、同分布数据的假设。

---

## 📊 项目概述

后门攻击利用模型训练期间的漏洞，在触发时诱导特定行为。为了对抗这些威胁，通常采用依赖一小部分称为辅助数据的干净数据集的后门净化技术。尽管取得了进展，但辅助数据特征对净化效果的影响仍研究不足。本项目探究了从同分布到合成或外部来源的不同类型的辅助数据集如何影响净化结果，旨在深入研究如何构建现实中有效的后门防御机制。

[概览](./overview.png)

---

## 🛠️ 开始使用

请按照以下步骤设置项目：

1. **克隆仓库**
    ```bash
    git clone https://github.com/shawkui/BackdoorBenchER.git
    cd BackdoorBenchER
    ```

2. **安装依赖**
    ```bash
    bash sh/install.sh
    ```

3. **初始化文件夹**
    ```bash
    bash sh/init_folders.sh
    ```

---

## ⚙️ 使用说明

### 🧪 创建辅助数据集

例如，对于CIFAR-10：

1. 将数据集下载到`/data`。
2. 分割数据：
    ```bash
    python dataset/generate_split.py --dataset cifar10 --split_ratio 0.05 --random_seed 0
    ```
3. 生成OOD辅助数据：
    ```bash
    python dataset/generate_ood.py --dataset cifar10_split_5_seed_0 --ood_type brightness
    ```
4. 从ImageNet创建类似CIFAR-10的数据集：
    ```bash
    bash sh/cinic_download.sh
    python dataset/generate_cifar10_from_imagenet.py --dataset cifar10_split_5_seed_0 --ood_type imagenet
    ```

### 🛡️ 执行攻击与防御
模拟攻击：
```bash
python attack/badnet.py --save_folder_name badnet_demo --dataset cifar10_split_5_seed_0
```
应用防御：
```bash
python defense/ft.py --result_file badnet_demo --dataset cifar10_split_5_seed_0 --reserved_type reserved 
```

通过编辑`sh/config_edit.py`自定义所有方法的配置。

### 📄 管理结果
根据`--yaml_path`参数中指定的配置保存所有的防御结果。

例如，

```bash
python defense/ft.py --result_file badnet_demo --dataset cifar10_split_5_seed_0 --reserved_type reserved --yaml_path ./config/defense/ft/demo.yaml
```
将结果保存在`record/badnet_demo/defense/ft/demo/ `

---

## 📋 TODO

📅 **即将发布的功能：**

1. **发布生成合成数据的代码**：我们将很快提供生成合成辅助数据集的代码，扩展可用于测试和评估的数据集种类。
   
2. **发布数据集**：除了代码，我们还计划发布一个专门为后门净化研究设计的精选数据集。

3. **引导输入校准工具**：推出首个尝试将辅助数据集与分布内数据集对齐的工具，以促进更有效的后门净化。

4. **扩展评估框架**：未来的长期更新将包括更全面的评估框架，涵盖更多的净化技术、模型和任务。我们欢迎相关科研工作者的积极参与。

敬请期待更新！

---

## 📄 引用

如果本项目对您的研究有帮助，请引用我们的工作：
```bibtex
@misc{wei2025revisitingauxiliarydatabackdoor,
      title={Revisiting the Auxiliary Data in Backdoor Purification}, 
      author={Shaokui Wei and Shanchao Yang and Jiayin Liu and Hongyuan Zha},
      year={2025},
      eprint={2502.07231},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2502.07231}, 
}
```

---

## 🎖️ 致谢

我们的工作基于[BackdoorBench](https://github.com/SCLBD/BackdoorBench)。如果他们的工作对您有帮助，请考虑给他们加星。

我们的工作建立在先前工作的基础上，包括但不限于：

* https://github.com/SCLBD/BackdoorBench
* https://github.com/AISafety-HKUST/Backdoor_Safety_Tuning
* https://github.com/BayesWatch/cinic-10
* https://github.com/shawkui/Shared_Adversarial_Unlearning

---

## 📞 联系方式

如有疑问或反馈，请开一个问题或发送邮件至`shaokuiwei@link.cuhk.edu.cn`。



