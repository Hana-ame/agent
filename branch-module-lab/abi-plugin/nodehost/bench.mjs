// Node 侧基准：同一个 C ABI .so，经 N-API addon 调用。
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

const tLoad = process.hrtime.bigint();
const addon = require("../build/greeter.node"); // Init 时 dlopen 共享库
const loadMs = Number(process.hrtime.bigint() - tLoad) / 1e6;

console.log(`[node] addon + dlopen 耗时: ${loadMs.toFixed(2)} ms（一次性）`);
console.log(`[node] name=${addon.name()} interface_version=${addon.interfaceVersion()} add(1,2)=${addon.add(1, 2)}`);
console.log(`[node] run("world") => ${addon.run("world")}`);

const N = 1_000_000;
let sum = 0;
let t = process.hrtime.bigint();
for (let i = 0; i < N; i++) sum += addon.add(1, 2);
const napiNs = Number(process.hrtime.bigint() - t) / N;

let jsSum = 0;
t = process.hrtime.bigint();
for (let i = 0; i < N; i++) jsSum += 1 + 2;
const jsNs = Number(process.hrtime.bigint() - t) / N;

t = process.hrtime.bigint();
addon.benchSum(N); // 一次跨语言调用，C 里算 N 次
const batchNs = Number(process.hrtime.bigint() - t) / N;

console.log("[node] 基准（本机，仅相对比较）:");
console.log(`[node]   pure JS add       : ${jsNs.toFixed(2)} ns/op`);
console.log(`[node]   N-API → C add     : ${napiNs.toFixed(1)} ns/op  （每次跨语言边界）`);
console.log(`[node]   1 次调用算 1e6 次 : ${batchNs.toFixed(2)} ns/element（批量摊薄后）`);
void sum;
void jsSum;