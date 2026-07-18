---
layout: home

hero:
  name: "Advanced RAG"
  text: "工业级双轨检索与自适应语义降噪引擎"
  tagline: "解决超长学术/技术文档边缘部署的幻觉与显存瓶颈"
  image:
    src: https://vitepress.dev/vitepress-logo-large.png
    alt: VitePress Logo
  actions:
    - theme: brand
      text: 快速开始
      link: /intro
    - theme: alt
      text: 核心算法介绍
      link: /intro#🏗️-核心算法

features:
  - icon: 🧠
    title: 语义锚点父子替换
    details: 检索时使用精细 Child 块计算相似度，召回后自动映射并替换为 Parent 父块，兼顾高召回率与连贯上下文。
  - icon: 🛡️
    title: 自适应语义断崖阻断
    details: 在重排精排阶段动态计算相邻分值落差，一旦发生断崖下跌即时切断，拦截达 50% 的无关噪点。
  - icon: ⚡
    title: 线程池双通道并发
    details: 并发调用 Chroma 向量与 BM25 关键词双检索通道，多路检索时延压缩至最大单路耗时。
---
