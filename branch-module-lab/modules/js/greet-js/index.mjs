// JS 模块：默认导出统一接口 { name, describe, run }
export default {
  name: "greet-js",
  describe() {
    return "JS ESM 模块：生成问候语（v3）";
  },
  run(payload) {
    return `hello, ${payload} (js v3)`;
  },
};
