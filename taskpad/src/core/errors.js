'use strict';

/**
 * 业务错误：领域层只负责给出「状态码 + 可读原因」，
 * 由 router 统一转成 JSON 响应，领域层不碰 req/res。
 */
class AppError extends Error {
  constructor(status, message, detail) {
    super(message);
    this.name = 'AppError';
    this.status = status || 500;
    this.detail = detail === undefined ? null : detail;
  }
}

const badRequest = (msg, detail) => new AppError(400, msg, detail);
const notFound = (msg, detail) => new AppError(404, msg, detail);
const conflict = (msg, detail) => new AppError(409, msg, detail);
const forbidden = (msg, detail) => new AppError(403, msg, detail);

module.exports = { AppError, badRequest, notFound, conflict, forbidden };
