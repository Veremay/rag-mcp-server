---
name: checkpoint
description: Summarize completed work, update progress tracking in DEV_SPEC.md, and prepare for next iteration. Final stage of dev-workflow pipeline. Use when task implementation and testing is completed, or when user says "完成检查点", "checkpoint", "保存进度", "save progress", "任务完成".
metadata:
  category: progress-tracking
  triggers: "execute_checkpoint_flow, archive_checkpoint"
allowed-tools: Bash(python:*) Bash(git:*) Read Write
---

# Progress Persistence (Interactive Flow)

## 🛑 Anti-Loop Guardrails (CRITICAL)

To prevent recursion loops, you **MUST** follow these rules:

1.  **INTERACTIVE MODE ONLY**: This skill requires multiple user turns. **NEVER** attempt to complete the entire flow in a single response.
2.  **MANDATORY STOP**: After outputting a verification summary or a confirmation prompt, you **MUST TERMINATE YOUR TURN IMMEDIATELY**. Do not output any further text or call tools until the user replies.
3.  **NO SIMULATION**: Never hallucinate or simulate a user's "confirm" response. You must wait for the actual user to type "confirm" or "yes".
4.  **NO RECURSION**: Do not call the `checkpoint` skill tool again from within this flow. Use standard tools (`Read`, `Write`, `SearchReplace`, `RunCommand`) to perform actions.
5.  **SAFE OUTPUT**: Do not use phrases like "TASK COMPLETED" (all caps) in your output, as this may trigger system automations. Use "Task Processed" instead.

---

## Workflow Overview

This process is split into **3 distinct interaction turns**. You must stop and yield to the user after each step.

```
Turn 1: Assistant generates SUMMARY → User VERIFIES
Turn 2: Assistant UPDATES SPEC & PROPOSES COMMIT → User APPROVES
Turn 3: Assistant EXECUTES COMMIT → Done
```

---

## Turn 1: Work Summary & Verification

**Goal**: Generate a summary of work done and ask the user to verify it.

### Actions
1.  **Gather Context**: Check recent file changes, test results, and the active task in `DEV_SPEC.md`.
2.  **Generate Report**: Output the summary using the format below.
3.  **STOP**: End your turn immediately.

### Output Template (Turn 1)
```
════════════════════════════════════════════════════
 Please Verify Completion Summary / 请验证工作总结
════════════════════════════════════════════════════

 Task: [Task ID] [Task Name]
 Spec Reference: DEV_SPEC.md Section [X.Y]

 Files Changed:
  Created:
    - src/...
  Modified:
    - src/...

 Test Results:
    - [Test File]: [Pass/Fail]
    - Coverage: [XX%]

 Iterations: [N]

════════════════════════════════════════════════════
 Is this summary accurate?
 以上总结是否准确？

   Please reply: "confirm" / "确认" to proceed with progress update.
                "revise" / "修改" to regenerate summary.
════════════════════════════════════════════════════
```

> **ACTION**: STOP HERE. Do not proceed to Turn 2.

---

## Turn 2: Persist Progress & Commit Prep

**Trigger**: User says "confirm", "yes", "确认", or "是".

**Goal**: Update `DEV_SPEC.md` and generate the git commit message.

### Actions
1.  **Update DEV_SPEC.md**: Use `SearchReplace` to update **ALL THREE** progress indicators in the GLOBAL `DEV_SPEC.md` file. You may need multiple `SearchReplace` calls.
    -   **Update 1: Phase Table** (e.g., `| I1 | ... | [ ] | ...`): Change `[ ]` to `[x]` (or `✅`) and fill in the "Complete Date" column with today's date.
    -   **Update 2: Overall Progress** (e.g., "总体进度"): Increment the "Completed" count for the current Phase and the Total. Recalculate and update the "Progress" percentage.
    -   **Update 3: Task Header** (e.g., `### I1：Task Name`): Append ` ✅` to the end of the task header line.
2.  **Generate Commit Message**: Create a conventional commit message based on the work.
3.  **Ask for Approval**: Present the message and ask if you should run `git commit`.
4.  **STOP**: End your turn immediately.

### Output Template (Turn 2)
```
────────────────────────────────────
 DEV_SPEC.md Updated
────────────────────────────────────
 Task: [Task ID] [Task Name]
 Status: Marked as Completed [x]
────────────────────────────────────

════════════════════════════════════════════════════
 PROPOSED COMMIT / 建议提交
════════════════════════════════════════════════════

 command: git commit -m "..."

 Message:
 [Subject]
 feat(scope): [Phase X.Y] summary

 [Description]
 ...

════════════════════════════════════════════════════
 Do you want to execute this commit?
 是否执行提交？

   Reply: "yes" / "commit" / "是" to execute.
          "no" / "skip" / "否" to skip commit.
════════════════════════════════════════════════════
```

> **ACTION**: STOP HERE. Do not proceed to Turn 3.

---

## Turn 3: Execute Commit

**Trigger**: User says "yes", "commit", "确认", or "是".

**Goal**: Run the git commands to save changes.

### Actions
1.  **Execute Commands**:
    ```bash
    git add .
    git commit -m "subject" -m "description"
    ```
2.  **Report Success**: Confirm the commit hash and branch.

### Output Template (Turn 3)
```
────────────────────────────────────
 COMMIT SUCCESSFUL
────────────────────────────────────
 Commit: [Short Hash]
 Branch: [Branch Name]

 Task [Task ID] workflow completed!
 任务 [Task ID] 流程已完成！
────────────────────────────────────
```

---

## Quick Reference

| User Intent | Correct Action |
| :--- | :--- |
| **"Check progress"** | Run Turn 1 (Summary) |
| **"Confirm summary"** | Run Turn 2 (Update Spec + Prep Commit) |
| **"Do commit"** | Run Turn 3 (Execute Commit) |

**Remember**: The most common cause of loops is trying to do Turn 1 and Turn 2 in the same response. **ALWAYS STOP** after asking a question.
