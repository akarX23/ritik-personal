---
name: git-commit
description: "Create safe, context-aware commit messages for this MNIST project using git status and git diff HEAD. Use when preparing commits, especially for model architecture or training loop changes. Uses git commands in bash only and never pushes automatically."
argument-hint: "Optional: commit intent, for example fix overfitting or refactor training loop"
user-invocable: true
---

# Git Commit Skill for MNIST

## Purpose

Create a high-quality commit from current repository changes by:
- Gathering dynamic context from git status and git diff HEAD
- Prioritizing model and training loop changes in the commit message
- Requesting explicit user approval before creating the commit
- Committing only with git commands in bash
- Never pushing automatically

## Scope

This skill is project-scoped for this repository and should be used from the repository root.

## Allowed Execution Surface

Use only bash terminal execution with git commands and basic shell operators for command chaining.

Allowed examples:
- git status
- git diff HEAD
- git diff --stat HEAD
- git diff --name-only HEAD
- git add
- git commit
- git restore --staged

Disallowed actions:
- Any push command
- Any non-git tooling for commit generation
- Any automatic commit without explicit user approval in-session

## Procedure

1. Confirm repository context.
- Run git rev-parse --show-toplevel
- If outside repo, stop and report

2. Collect dynamic context.
- Run git status --porcelain=v1 --branch
- Run git diff --stat HEAD
- Run git diff --name-only HEAD
- Run git diff HEAD

3. Detect change focus with priority.
- Highest priority: model definition or architecture updates
  - Typical signals: files or hunks touching model, network, layer, module, forward, encoder, decoder
- Next priority: training loop logic updates
  - Typical signals: files or hunks touching train, epoch, batch, optimizer, scheduler, loss, backward, step, dataloader
- Next: evaluation or metrics logic
- Last: docs, comments, or housekeeping

4. Draft commit message.
- Message should reflect the highest-priority change detected.
- Prefer concise format:
  - type(scope): summary
- Scope guidance:
  - model for architecture changes
  - train for training loop and optimization changes
  - eval for evaluation logic
  - data for data pipeline
  - docs for documentation
  - chore for non-functional maintenance
- Summary guidance:
  - Use a verb and specific intent, for example improve, fix, refactor, tune, add
  - Mention model or training loop explicitly when applicable

5. Present summary and request approval.
- Provide:
  - detected focus
  - affected files (concise)
  - proposed commit message
- Ask explicit approval before running git add or git commit.

6. Commit after approval only.
- Stage intended files with git add -A unless user requests a narrower scope.
- Create commit using the approved message.
- Return resulting commit hash and one-line summary.

7. Post-commit safety checks.
- Run git status --porcelain=v1 --branch
- Confirm no push was attempted.
- Do not run git push under any circumstance unless user asks in a separate explicit request.

## Decision Logic

- If no changes are present, do not commit; report no-op.
- If only docs or formatting changed, use docs or chore scope.
- If both model and training loop changed, prefer model scope and mention training in summary when concise.
- If intent from user argument conflicts with detected diff, prefer detected diff and surface the mismatch before approval.

## Quality Checklist

Before committing, verify all are true:
- Dynamic context collected from git status and git diff HEAD
- Commit message references the most impactful change
- Model or training loop changes are explicitly referenced when present
- Explicit user approval captured in-session
- No push command executed

## Example Prompts

- Use git-commit to commit current changes with focus on training loop updates.
- Use git-commit with intent: fix model regularization behavior.
- Use git-commit and include only staged changes.
