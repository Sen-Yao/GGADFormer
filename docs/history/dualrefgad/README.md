# DualRefGAD 历史资料迁移说明

迁移日期：2026-07-20

本目录保存从 `/Users/oliver/Developer/DualRefGAD` 复制的 VecGAD 相关历史资料。原件未删除、未修改。

## 已复制内容

| 来源 | 当前目录 | 文件数 | 用途 |
|---|---|---:|---|
| `baseline/vecgad/` | `baseline-notes/` | 6 | VecGAD 方法、protocol、Photo 信号及 DualRefGAD 对照卡 |
| `investigations/2026-07-01-dualrefgad-vecgad-style-residual-hard-negative-probe/` | `2026-07-01-source-probe/` | 6 | 源码机制阅读、假设、结论和活动记录 |
| `literature/vecgad/` | 仓库根目录 `literature/vecgad/` | 4 | 本地论文 PDF、metadata 和阅读卡 |

复制后逐文件执行了 SHA-256 校验，所有目标文件与原件一致。KDD PDF 的 SHA-256 为：

```text
2c68c240cc01edccf7f4c6ce10f746365316deda90664c624eb3329db4811560
```

## 未重复复制的内容

历史 investigation 中的 `references/code/GGADFormer/` 没有复制到本目录。它是 VecGAD GitHub 仓库在提交 `28bce1a83bc87d7cd1d2dce423da7c79b296c5b7` 的 clone，与当前根目录的 canonical code 完全相同。保留第二个嵌套 Git 仓库只会制造来源歧义。

## 使用边界

这些文件是原样保存的历史记录，正文可能使用 DualRefGAD 的问题设定和术语。它们可以作为机制阅读与研究谱系证据，但不能覆盖当前 VecGAD 代码、论文或正式实验 protocol。
