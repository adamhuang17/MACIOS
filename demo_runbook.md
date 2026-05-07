# Agent-Hub 比赛演示 Runbook（黄金路径版）

> 本 Runbook 描述当前已手动验证的演示链路：在飞书发起“生成带内容的 PPT”→ Agent 规划 → 用户审批 → 多端实时同步 → 飞书返回可打开的 `deck.pptx` → 电脑/手机均可打开。
> 演示中遇到异常以本文“兜底路径”和“常见问题”兜底，不要临场重写脚本。

---

## 0. 一句话价值

> Agent-Hub 是一个“多端可控、可审批、可恢复”的 Agent 协作中台：
> 用户在飞书一句话提需求，Agent 自动规划成多个步骤，关键动作（如生成、外发分享）走人审批；执行过程在飞书消息、桌面 Dashboard、手机端三端实时同步；最终交付一份可在 Office 直接打开的 `.pptx` 共享文件。

---

## 1. 演示前 5 分钟自检

### 1.1 配置检查（`.env`）

| 必填项 | 含义 | 备注 |
| --- | --- | --- |
| `FEISHU_ENABLED=true` | 启用飞书连接器 | |
| `FEISHU_USE_LONG_CONN=true` | 使用长连接 | 无公网回调环境必选 |
| `FEISHU_APP_ID` / `FEISHU_APP_SECRET` | 飞书自建应用密钥 | |
| `FEISHU_BOT_OPEN_ID` | 机器人 open_id | 群聊 `@bot` 必需 |
| `FEISHU_REQUIRE_MENTION_IN_GROUP=true` | 群聊必须 `@bot` | 演示群推荐打开 |
| `PILOT_ENABLED=true` | 启用 Pilot 编排 | |
| `PILOT_STORE_PATH=./data/pilot.sqlite3` | 任务持久化 | 演示用固定路径 |
| `PILOT_SKILL_MODE=real_chain` | 真实 PPT 链路 | 演示必选；fake 仅用于测试 |
| `PILOT_PROGRESS_HEARTBEAT_INTERVAL_SECONDS=5` | 长任务心跳写入频率 | Dashboard 显示“最近活动 5 秒前” |
| `PUBLIC_BASE_URL=http://127.0.0.1:8080` | Dashboard URL 模板 | 局域网演示改本机 IP |
| `FEISHU_DRIVE_FOLDER_TOKEN` | 上传到的飞书云空间 folder | 已验证可用 |
| `FEISHU_ADMIN_OPEN_ID` | 演示账号 open_id | 用于自动加协作权限 |
| `FEISHU_PROGRESS_MIN_INTERVAL_SECONDS=5` | 飞书进度消息最小间隔 | 防止长任务心跳刷屏 |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | LLM 网关 | 演示用 OpenAI 兼容端点 |

### 1.2 数据准备

- 清理或重命名旧的 `data/pilot.sqlite3`（保留也行，但建议演示前用一个干净 workspace）。
- 确认 `data/pilot/artifacts/` 可写。
- 飞书演示账号已加入演示群、机器人已加入演示群，并具备“接收消息/发消息/上传云文档”权限。

### 1.3 启动服务

```powershell
E:/Anaconda/envs/Agent-Hub/python.exe -m uvicorn agent_hub.api.routes:app --host 0.0.0.0 --port 8080
```

启动后看以下日志中存在即认为 OK：

- `pilot_runtime.store kind=sqlite`
- `feishu.runtime.long_conn_enabled`
- `feishu.longconn.started`
- `feishu_long_conn_started`
- `pilot_dashboard_mounted`
- `api_started`

打开 Dashboard：`http://127.0.0.1:8080/dashboard/`（手机连同一网，输入 `http://<本机IP>:8080/dashboard/`）。

### 1.4 自动化基线（可选）

```powershell
E:/Anaconda/envs/Agent-Hub/python.exe scripts/offline_verify.py
E:/Anaconda/envs/Agent-Hub/python.exe -m pytest tests -q -x
```

---

## 2. 黄金路径（主演示，约 3 分钟）

### 步骤 1 · 飞书发起任务

在演示群（或私聊机器人）发送：

> @Agent-Hub 现在帮我生成一份带有内容的 Agent-Hub 项目路演 PPT。

预期：
- 服务端日志：`feishu.ingress.classified intent=start_task` → `feishu.webhook.task_submitted`。
- 飞书机器人立即回执“收到任务，开始执行，正在规划：…”；随后可看到任务编排/执行进度消息。
- Dashboard 左侧任务列表新增一条任务，状态为 **规划中** 或 **等待审批**。
- Dashboard 顶部活动条会显示“收到任务，开始执行”“Agent 正在进行任务编排”或“Agent 仍在工作，最近活动 5 秒前”。

### 步骤 2 · 三端同步审批

待 Agent 完成规划后：
- **飞书**：群里出现“执行计划审批”交互卡片，可直接点【同意】/【拒绝】。
- **桌面 Dashboard**：选中任务，顶部出现黄色“等待审批”活动条，下方“待办审批”卡片提供同等按钮。
- **手机 Dashboard**：用手机浏览器打开 Dashboard URL，菜单按钮收起任务列表，主区可看到同一审批卡片，按钮足够大。

演示话术示例：
> “你看，飞书、电脑端、手机端是同一个 task 状态——任意一端审批，其它两端都会立即更新。”

任意一端点【同意】。预期：
- 三端的“等待审批”立即消失。
- Dashboard 顶部活动条切换为“正在 生成 PPT 大纲 / 渲染 PPT 文件 / 上传飞书云文档并分享”等中文文案。
- 步骤时间线上对应步骤切换为 **执行中** → **已完成**。

### 步骤 3 · 实时执行可视化

讲解期间观察 Dashboard：
- 步骤区按真实顺序逐个亮起：收集上下文 → 生成需求 Brief → 生成 PPT 大纲 → 渲染 PPT 文件 → 上传飞书云文档并分享 → 生成总结。
- “最近事件”区域实时刷新，可指着说明：plan.generated → step.status_changed → artifact.created → artifact.status_changed=shared。
- “最近事件”里会持续出现 `task.progress`，Dashboard 和飞书端看到的是同一条持久化进度事实流。
- 顶部右上角“实时连接”绿点表示 SSE 仍在线。

### 步骤 4 · 飞书返回成果

任务成功后：
- 飞书群里 Agent 回复一段简短结果消息，并附带 `deck.pptx` 文件卡片（飞书云文档分享）。
- **电脑端**：点击该文件 → 浏览器跳转飞书云文档；点击下载或在 Office 打开均可。
- **手机端**：在飞书 App 内点击文件 → 选择“用其他应用打开”→ Office 可正常预览。
- Dashboard 的“成果”区域显示：
  - 📝 需求 Brief（可点【预览】展开 markdown）
  - 🗂️ PPT 大纲（可点【预览】查看 SlideSpec JSON）
  - 📊 PPT 文件（显示页数、版本）
  - ☁️ 飞书云文档（提供【打开分享】【复制链接】按钮）
  - ✅ 任务总结（可预览）

### 步骤 5 · 收尾话术

> “整个链路从一句自然语言开始，Agent 自主规划、人在关键点审批、三端实时一致、最终交付一个可在 Office 打开的 PPT 文件——这就是我们对‘多端协同 Agent 中台’的解法。”

---

## 3. 兜底路径

| 场景 | 兜底操作 | 预期结果 |
| --- | --- | --- |
| 飞书卡片审批未生效 | 在 Dashboard“待办审批”里点【同意】 | 任务继续执行，飞书卡片随后变更 |
| 长连接断开 | Dashboard 顶部红点提示 → 重启服务 | 选中任务后 SSE 自动重连，事件回填 |
| PPT 渲染失败 | Dashboard 步骤区显示错误，点【重试任务】或【从某步恢复】 | 生成新的恢复任务从失败步骤继续 |
| Drive 上传失败 | 成果区仍能看到本地 PPTX，飞书消息会带本地 artifact 提示 | 演示时改讲“本地交付兜底” |
| 飞书 LLM 误判普通问答为 ignore | 新版会自动 promote 为 `llm_agent` 回复 | 日志可见 `pipeline.addressed_ignore_promoted` |

---

## 4. 加分演示（备用，时间充足时再加）

- **进度查询**：`@Agent-Hub 刚才那个 PPT 任务进度怎么样？`
  日志出现 `intent=progress_query`，飞书返回当前状态。
- **重试演示**：临时改环境让某步失败 → 在 Dashboard 演示一键“恢复”能力。
- **手机审批演示**：让评委用自己的手机扫一下 Dashboard 二维码，实际点一次审批，强化“多端”。

---

## 5. 已验证 / 暂不主推

**已手动验证（可放心讲）**

- 飞书私聊/群聊触发任务、卡片审批、最终消息附带可打开的 `deck.pptx`。
- 桌面 Dashboard 和手机 Dashboard 同步显示任务状态、审批、成果。
- Dashboard 移动端单列布局、操作按钮可点、成果区可点【打开分享】/【复制链接】。
- SSE 实时事件流、SQLite 持久化、artifact 文件落盘。

**暂不主推（避免临场翻车）**

- 多人离线编辑自动合并：当前是“弱网兜底 + 本地状态恢复”，不是“离线协同编辑”。
- 复杂 PPT 设计/自由画布：当前 PPT 内容偏标准化叙事，不要承诺“高级设计器”。
- Webhook 公网模式：演示用长连接更稳；如确需公网回调，提前 30 分钟切到 webhook 模式回归一遍。

---

## 6. 故障速查

| 现象 | 重点日志 | 优先排查 |
| --- | --- | --- |
| 飞书无任何反应 | `feishu.longconn.started` 是否出现 | 长连接开关、`APP_ID/SECRET`、`lark-oapi` 安装 |
| 群聊 @ 后无反应 | `reason=group_without_mention` | `@bot` 是否真的命中、`FEISHU_BOT_OPEN_ID` 是否对 |
| 普通问答无回复 | `pipeline.addressed_ignore_promoted` / `feishu.ingress.reply_failed` | 服务是否重启到最新版、LLM 网关 key |
| Dashboard 没任务 | `pilot_dashboard_mounted`、`PILOT_STORE_PATH` | 是否打开 `/dashboard/`、是否 SQLite 模式 |
| 审批后任务不动 | `approval.decided` 后是否有 `step.status_changed` | 卡片是否被多次决议、查看 Dashboard“可执行操作”里有无【继续执行】 |
| PPT 无法打开 | 飞书消息里给的是分享链接还是本地 path | 确认 `real.drive.upload_share` 步骤成功、Drive folder/admin open_id 配置 |
