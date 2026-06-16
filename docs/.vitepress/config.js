import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(defineConfig({
  title: "Advanced RAG Docs",
  description: "Documentation for Advanced RAG Engine",
  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '指南', link: '/intro' }
    ],
    sidebar: [
      {
        text: '高级 RAG 指南',
        items: [
          { text: '项目介绍', link: '/intro' }
        ]
      },
      {
        text: '🤖 PoiClaw Agent 重构实战',
        items: [
          { text: '1. 前言与本地模型适配', link: '/poiclaw/intro' },
          { text: '2. 状态机与 Workflow 剖析', link: '/poiclaw/workflow' },
          { text: '3. 手写 JSON 容错与 Pytest 单元测试', link: '/poiclaw/parser' },
          { text: '4. 命令执行沙箱与敏感词过滤', link: '/poiclaw/sandbox' },
          { text: '5. 长时间任务（Goal Loop）大闭环', link: '/poiclaw/goal' }
        ]
      }
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com' }
    ],
    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2026-present 尖子'
    }
  }
}))
