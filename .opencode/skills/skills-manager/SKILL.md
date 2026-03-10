---
name: skills-manager
description: Provides comprehensive overview and management of all installed Claude skills. Use when users ask to list skills, see what skills are available, understand skill functionality, or manage their skill collection. Triggered by queries like "what skills do I have", "list all skills", "show me my skills", or any request to browse, search, or understand available capabilities.
---

# Skills Manager

Provides a comprehensive overview of all installed skills, including their names, descriptions, and bundled resources. This skill helps users discover and understand available capabilities.

## Quick Start

To list all available skills with their descriptions:

```bash
~/.claude/skills/skills-manager/scripts/list_skills.py
```

This outputs:
- Skill names and directories
- Descriptions of when each skill triggers
- Resources included (scripts, references, assets)

## Understanding Skill Resources

Skills may include three types of bundled resources:

**📜 Scripts** - Executable code (Python/Bash) for automation and operations
**📚 References** - Documentation loaded into context when needed
**🎨 Assets** - Templates, boilerplate, or files used in output

## Usage Patterns

Use this skills-manager when:
- User asks "what skills do I have?" or "list all skills"
- User wants to discover available capabilities
- User needs to understand what a specific skill does
- User is exploring which skills might help with a task

The list_skills.py script parses SKILL.md frontmatter from all skill directories and presents a formatted overview.
