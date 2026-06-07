#!/bin/sh
# Git 便捷脚本 — 绕过鸿蒙沙箱 exec 限制
# 用法: ./git.sh <command> [args...]
#
# 示例:
#   ./git.sh push             推送当前分支
#   ./git.sh push --force     强制推送
#   ./git.sh push main        推送指定分支
#   ./git.sh status           查看状态
#   ./git.sh log              查看日志
#   ./git.sh sync             推送 + 更新 tracking ref

GIT_DIR="$(cd "$(dirname "$0")" && pwd)"
PUSH_SCRIPT="$GIT_DIR/.git/push.js"

case "$1" in
  push)
    shift
    FORCE=""
    BRANCH=""
    REMOTE=""
    for arg in "$@"; do
      case "$arg" in
        --force|-f) FORCE="--force" ;;
        *)
          if [ -z "$BRANCH" ]; then BRANCH="$arg"
          elif [ -z "$REMOTE" ]; then REMOTE="$arg"; fi
          ;;
      esac
    done
    # 获取当前分支
    [ -z "$BRANCH" ] && BRANCH=$(git rev-parse --abbrev-ref HEAD)
    echo "==> 推送 $BRANCH${FORCE:+ (强制)}..."
    node "$PUSH_SCRIPT" "$BRANCH" "$REMOTE" $FORCE
    if [ $? -eq 0 ]; then
      # 更新本地 tracking ref
      LOCAL=$(git rev-parse HEAD)
      git update-ref "refs/remotes/origin/$BRANCH" "$LOCAL" 2>/dev/null
      echo "==> Tracking ref 已更新"
    fi
    ;;
  sync)
    shift
    ./git.sh push "$@"
    ;;
  status|st)
    git status
    ;;
  log|l)
    git log --oneline --graph -20
    ;;
  diff|d)
    git diff
    ;;
  commit|c)
    shift
    git commit "$@"
    ;;
  add)
    shift
    git add "$@"
    ;;
  branch|b)
    git branch -a
    ;;
  fetch)
    echo "fetch 需要 SSH, 暂不支持。使用 ./git.sh push 推送后手动更新 ref。"
    ;;
  pull)
    echo "pull 需要 SSH, 暂不支持。"
    ;;
  help|--help|-h)
    echo "Git 便捷脚本 - 鸿蒙沙箱适配版"
    echo "用法: ./git.sh <command> [args]"
    echo ""
    echo "可用命令:"
    echo "  push [分支] [远程] [--force]  推送代码"
    echo "  sync [分支] [远程] [--force]  推送 + 同步"
    echo "  status (st)                   查看状态"
    echo "  log (l)                       查看日志"
    echo "  diff (d)                      查看差异"
    echo "  commit (c)                    提交"
    echo "  add                           添加文件"
    echo "  branch (b)                    查看分支"
    ;;
  *)
    git "$@"
    ;;
esac
