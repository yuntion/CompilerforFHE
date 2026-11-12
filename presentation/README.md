# 毕业设计近期进度汇报

- `progress.tex`：LaTeX Beamer 源码，16:9，共 20 页。
- `progress.pdf`：编译后的演示文稿。
- 风格依据：根目录 `20261028.pptx` 的深蓝、浅蓝、青绿配色与章节页、卡片布局。

编译（需要 XeLaTeX、ctex、Beamer、TikZ，以及 Windows 常见字体宋体 SimSun、微软雅黑 Microsoft YaHei、仿宋 FangSong）：

```bash
cd presentation
xelatex -interaction=nonstopmode -halt-on-error progress.tex
xelatex -interaction=nonstopmode -halt-on-error progress.tex
```

内容核对以根 README、front/backend 文档、docs/VALIDATION.md、fhe-coder 实际源码和 main.pdf 为依据。各内容页附材料路径。

历史测试记录与安全配置单次测量均有注明，并非本次重新测试。未将 fhe-coder README 的上游榜单作为毕设实验结果。当前 PDF 中方法与实验章节仍待补充，因此将其列为后续写作任务；AI 正式联调和四周排期属于拟开展工作。

封面姓名、导师取自 main.pdf 英文扉页，日期为本次制作日期，可在源码导言区修改。

第 13–14 页补充“如何使用 AI 生成后端代码”的回答与加乘示例；完整口头回答见 `AI-backend-answer.md`。这些页面说明拟议联调方式，不表示已完成自动生成验证链路。
