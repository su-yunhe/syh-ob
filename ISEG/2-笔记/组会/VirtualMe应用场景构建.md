# 1. 研发问答QA

当研发问答插件认为当前用户的输入不够，需要依赖用户行为数据进行知识增强时，就会通过@VirtualMe命令调用后端预处理、模式识别、日志总结、意图预测和能力分析中的一个或多个模块，并获取相应数据。
## 场景1-隐式依赖
用户操作涉及**跨文件、跨会话**的复杂关联，需历史行为串联。
### 开发者输入：

```
为什么从 `utils/logger.js` 导入 `logError` 报错？
```

### 知识增强前的输出：
```
请检查导入路径是否正确，例如：  
import { logError } from './utils/logger';  
```

### 存在问题：
未发现用户修改了`logger`模块导出方式。

### 知识增强的过程：

研发问答：
```
@VirtualMe，日志总结。
```

VirtualMe：
1. 工作模式：编写代码；
2. 聚焦位置：logger.js；
3. 行为总结：
	 - 重点编辑的源代码：

```
// logger.js
function logInfo(message) {
    console.log("Info:", message);
}

function logError(error) {
    console.error("Error:", error);
}

export default { logInfo, logError };
```

	 - 重点编辑过的代码变更摘要：

```
将 `logger.js` 的导出方式从命名导出改为默认导出
```

### 知识增强后的输出：
```
检测到您最近将 `logger.js` 的导出方式从命名导出改为默认导出。  
解决方案：  
1. 修改导入语句为：  
   import logger from './utils/logger';  
   logger.logError(...);  
2. 或恢复原导出方式：  
   export const logError = (...) => { ... };  
```


## 场景2-长期任务跟踪

### 开发者输入：
```
这个 `NullPointerException` 怎么还没解决？
```

### 知识增强前的输出：
```
请检查对象是否初始化，例如：  
if (obj != null) { ... }  
```

### 存在的问题：
未关联用户一周内的调试历史，建议过于笼统。

### 知识增强后的输出：
```
结合过去一周的调试记录，发现以下线索：  
1. 您曾修复 `UserService.fetchUser` 中类似的空指针问题（通过添加 `Optional` 包装）。  
2. 当前报错位于 `OrderService.calculateTotal`，该函数调用 `fetchUser` 但未处理 `null` 返回值。  
推荐方案：  
1. 在 `calculateTotal` 中添加空值检查：  
   if (user == null) throw new UserNotFoundException();  
2. 复用 `UserService` 的 `Optional` 模式：  
   Optional.ofNullable(user).orElseThrow(...);  
```


## 场景3: 个性化习惯（编码习惯、框架偏好等）
### 开发者输入：
```
如何实现表单验证？
```

### 知识增强前的输出：
```
推荐使用 Joi 或 Yup 进行校验：  
const schema = Joi.object({ name: Joi.string().required() });  
```

### 问题：
未适配用户习惯，推荐了开发者不熟悉的库。

### 知识增强后的输出：
```
检测到您的历史项目中频繁使用 React Hook Form + Zod。  
推荐方案：  
1. 使用 Zod 定义 Schema：  
   const schema = z.object({ name: z.string().min(1) });  
2. 集成 React Hook Form：  
   const { register, handleSubmit } = useForm({ resolver: zodResolver(schema) });  
```


## 场景4: 主动推荐
存在未显式提问的潜在需求

### 用户输入：
（无显式提问，但频繁切换至 `api.ts` 文件并悬停查看接口定义）

### 知识增强前的输出：
```
（无响应，因未触发自然语言提问）  
```

### 知识增强后的输出：
```
检测到您在过去 5 分钟频繁查看 `api.ts` 中的 `fetchUserData` 接口。  
是否需要以下帮助？  
1. 生成该接口的 Swagger 文档：  
   @Get("/user/:id")  
   async fetchUserData(@Param("id") id: string) { ... }  
2. 生成 Mock 数据：  
   { "id": "1", "name": "John Doe", "role": "admin" }  
```


# 2. 开发者能力评估

## 知识深度
衡量开发者是否深入理解某个领域，或者仅仅是搜索/复制代码。

- **新 API 适应能力**：检测 MouseHover 在不常见 API 上的时间。
- **代码粘贴 vs. 编写比率**：PasteText 事件 vs. EditTextDocument 自编写代码的比例。
- **文档阅读量**：开发者是否经常 OpenTextDocument 查看 README.md、docs/ 目录等文件。

## 开发效率

- **专注时间**：长时间 EditTextDocument 但无 ChangeTextDocument，代表持续专注开发。
- **任务切换频率**：短时间内多次 ChangeTextDocument、OpenTextDocument 代表多任务开发。


## 可视化展示

雷达图：
```
               ⬆ 代码编写能力（90）
   代码阅读能力（85）        调试能力（70）
      知识深度（60）      开发效率（95）   
```


时间趋势图：
```
 100 |        *   *
  80 |  *    *   *
  60 |  *   *
  40 | *  *
  20 |*
   0 |-------------------
    1月  2月  3月  4月  5月
```

任务切换热力图：
```
08:00 ███       代码编写
09:00 ███████   代码阅读
10:00 ██        代码调试
11:00 ██████    任务切换多
...
```

