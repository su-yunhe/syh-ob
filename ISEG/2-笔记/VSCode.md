# VSCode

## 1. 源码结构

### 目录结构
```
├── build         # gulp编译构建脚本
├── extensions    # 内置插件
├── resources     # 平台相关静态资源，图标等
├── scripts       # 工具脚本，开发/测试
├── out           # 编译输出目录
├── src           # 源码目录
├── test          # 测试套件
├── gulpfile.js   # gulp task
└── product.json  # App meta 信息
```

### src下的结构
```
├── bootstrap-amd.js    # 子进程实际入口
├── bootstrap-fork.js   #
├── bootstrap-window.js #
├── bootstrap.js        # 子进程环境初始化
├── buildfile.js        # 构建config
├── cli.js              # CLI入口
├── main.js             # 主进程入口
├── paths.js            # AppDataPath与DefaultUserDataPath
├── typings
│   └── xxx.d.ts        # ts类型声明
└── vs
    ├── base            # 通用工具/协议和基础 DOM UI 控件
    │   ├── browser     # 基础UI组件，DOM操作、交互事件、DnD等
    │   ├── common      # diff描述，markdown解析器，worker协议，各种工具函数
    │   ├── node        # Node工具函数
    │   ├── parts       # IPC协议（Electron、Node），quickopen、tree组件
    │   ├── test        # base单测用例
    │   └── worker      # Worker factory 和 main Worker（运行IDE Core：Monaco）
    ├── code            # vscode主窗体相关
    |   ├── electron-browser # 需要 Electron 渲染器处理API的源代码（可以使用 common, browser, node）
    |   ├── electron-main    # 需要Electron主进程API的源代码（可以使用 common, node）
    |   ├── node        # 需要Electron主进程API的源代码（可以使用 common, node）
    |   ├── test
    |   └── code.main.ts
    ├── editor          # 对接 IDE Core（读取编辑/交互状态），提供命令、上下文菜单、hover、snippet等支持
    |   ├── browser     # 代码编辑器核心
    |   ├── common      # 代码编辑器核心
    |   ├── contrib     # vscode 与独立 IDE共享的代码
    |   ├── standalone  # 独立 IDE 独有的代码
    |   ├── test
    |   ├── editor.all.ts
    |   ├── editor.api.ts
    |   ├── editor.main.ts
    |   └── editor.worker.ts
    ├── platform        # 支持注入服务和平台相关基础服务（文件、剪切板、窗体、状态栏）
    ├── workbench       # 协调editor并给viewlets提供框架，比如目录查看器、状态栏等，全局搜索，集成Git、Debug
    ├── buildunit.json
    ├── css.build.js    # 用于插件构建的CSS loader
    ├── css.js          # CSS loader
    ├── loader.js       # AMD loader（用于异步加载AMD模块，类似于require.js）
    ├── nls.build.js    # 用于插件构建的 NLS loader
    └── nls.js          # NLS（National Language Support）多语言loader
```

## 2. 技术架构

### 多进程架构

使用 **Electron** 架构，代码编辑器层为 Monaco。  

从实现上来看，Electron 的基本结构:  
> Electron = Node.js + Chromium + Native API

也就是说 Electron 拥有 Node 运行环境，依靠 Chromium 提供基于 Web 技术（HTML、CSS、JS）的界面交互支持，另外还具有一些平台特性，比如桌面通知。

从 API 设计上来看，Electron App 一般都有 1 个 Main Process 和多个 Renderer Process：  

- main process：主进程环境下可以访问 Node 及 Native API；
- renderer process：渲染器进程环境下可以访问 Browser API 和 Node API 及一部分 Native API。

VSC 采用多进程架构，VSC 启动后主要有下面的几个进程：
![[Pasted image 20241205142718.png]]

- 主进程
- 渲染进程
    - HTML 编写的 UI
        - Activitybar: 最左边(也可以设置到右边)的选项卡
        - Sidebar: Activitybar 选中的内容
        - Panel: 状态栏上方的面板选项卡
        - Editor: 编辑器部分
        - Statusbar: 下方的状态栏
    - Nodejs 异步 IO
        - FileService
        - ConfigurationService
- 插件宿主进程
- Debug 进程
- Search 进程

### 主进程

相当于后台服务，后台进程是 VSC 的入口，主要负责多窗体管理（创建/切换）、编辑器生命周期管理，进程间通信（IPC Server），自动更新，工具条菜单栏注册等。

我们启动 VSC 的时候，后台进程会首先启动，读取各种配置信息和历史记录，然后将这些信息和主窗口 UI 的 HTML 主文件路径整合成一个 URL，启动一个浏览器窗口来显示编辑器的 UI。后台进程会一直关注 UI 进程的状态，当所有 UI 进程被关闭的时候，整个编辑器退出。

此外后台进程还会开启一个本地的 Socket，当有新的 VSC 进程启动的时候，会尝试连接这个 Socket，并将启动的参数信息传递给它，由已经存在的 VSC 来执行相关的动作，这样能够保证 VSC 的唯一性，避免出现多开文件夹带来的问题。


### 渲染进程
编辑器窗口进程负责整个 UI 的展示，也就是我们所见的部分。UI 全部用 HTML 编写没有太多需要介绍的部分。

项目文件的读取和保存由主进程的 NodeJS API 完成，因为全部是异步操作，即便有比较大的文件，也不会对 UI 造成阻塞。IO 跟 UI 在一个进程，并采用异步操作，在保证 IO 性能的基础上也保证了 UI 的响应速度。

### 插件进程

每一个 UI 窗口会启动一个 NodeJS 子进程作为插件的宿主进程。所有的插件会共同运行在这个进程中。

### Debug 进程

Debugger 插件跟普通的插件有一点区别，它不运行在插件进程中，而是在每次 debug 的时候由 UI 单独新开一个进程。

### 搜索进程

搜索是一个十分耗时的任务，VSC 也使用的单独的进程来实现这个功能，保证主窗口的效率。将耗时的任务分到多个进程中，有效的保证了主进程的响应速度。


node版本：20.18.0

source ~/.nvm/nvm.sh