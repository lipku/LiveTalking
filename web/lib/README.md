# 第三方库本地化说明

本目录包含了所有从 CDN 下载的第三方库，以支持项目的离线部署。

## 目录结构

```
lib/
├── bootstrap/
│   ├── css/bootstrap.min.css (228 KB) - Bootstrap 5.3.0 CSS
│   └── js/bootstrap.bundle.min.js (79 KB) - Bootstrap 5.3.0 JS Bundle
├── bootstrap-icons/
│   ├── font/bootstrap-icons.css (94 KB) - Bootstrap Icons 1.10.0 CSS
│   └── fonts/
│       ├── bootstrap-icons.woff2 (119 KB) - 字体文件 (WOFF2)
│       └── bootstrap-icons.woff (161 KB) - 字体文件 (WOFF)
├── jquery/
│   ├── jquery-3.6.0.min.js (88 KB) - jQuery 3.6.0
│   └── jquery-2.1.1.min.js (83 KB) - jQuery 2.1.1
└── sockjs/
    └── sockjs-0.3.4.js (34 KB) - SockJS Client 0.3.4
```

## 使用的文件

### dashboard.html
- Bootstrap 5.3.0 (CSS + JS)
- Bootstrap Icons 1.10.0
- jQuery 3.6.0

### 其他 HTML 文件
以下文件使用 jQuery 2.1.1 和 SockJS 0.3.4：
- webrtcapi.html
- webrtcapi-asr.html
- webrtcapi-custom.html
- webrtcchat.html
- chat.html
- echo.html
- echoapi.html
- rtcpush.html
- rtcpushapi.html
- rtcpushapi-asr.html
- rtcpushchat.html
- webrtc.html

## 原始 CDN 源

所有资源均从以下 CDN 下载：
- Bootstrap: `https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/`
- Bootstrap Icons: `https://unpkg.com/bootstrap-icons@1.10.0/`
- jQuery: `https://code.jquery.com/`
- SockJS: `https://cdnjs.cloudflare.com/ajax/libs/sockjs-client/0.3.4/`

## 总大小

约 886 KB

## 更新说明

如需更新这些库，请：
1. 从对应的 CDN 下载最新版本
2. 替换本目录中的文件
3. 如果版本号变化，需要更新 HTML 文件中的引用路径

