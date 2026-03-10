#!/usr/bin/env python3
"""
Brainstorm Conversation Logger

Saves brainstorming conversations to markdown files in the argument directory.

Usage:
    uv run python save_conversation.py --topic "主题" --user "用户内容" --assistant "助手内容"
"""

import argparse
from datetime import datetime
from pathlib import Path


def get_argument_dir() -> Path:
    """Get the argument directory path (project root / argument)."""
    # Try to find project root by looking for common markers
    current = Path.cwd()

    # Walk up to find project root (contains .git or pyproject.toml)
    while current != current.parent:
        if (current / ".git").exists() or (current / "pyproject.toml").exists():
            return current / "argument"
        current = current.parent

    # Fallback: use current directory
    return Path.cwd() / "argument"


def format_entry(timestamp: str, user_content: str, assistant_content: str) -> str:
    """Format a conversation entry."""
    return f"""## {timestamp}

**用户**: {user_content}

**Claude**: {assistant_content}

---

"""


def create_new_file(filepath: Path, topic: str, entry: str) -> None:
    """Create a new conversation file with header."""
    header = f"""# {topic}

"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(header + entry, encoding="utf-8")
    print(f"✅ Created new file: {filepath}")


def append_to_file(filepath: Path, entry: str) -> None:
    """Append an entry to existing file."""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(entry)
    print(f"✅ Appended to: {filepath}")


def save_conversation(topic: str, user_content: str, assistant_content: str) -> Path:
    """
    Save a conversation entry to the argument directory.

    Args:
        topic: Topic name (used for filename)
        user_content: User's question or input
        assistant_content: Claude's response

    Returns:
        Path to the saved file
    """
    # Get argument directory
    arg_dir = get_argument_dir()

    # Create filename from topic
    filename = f"{topic}.qmd"
    filepath = arg_dir / filename

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Format entry
    entry = format_entry(timestamp, user_content, assistant_content)

    # Create or append
    if filepath.exists():
        append_to_file(filepath, entry)
    else:
        create_new_file(filepath, topic, entry)

    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Save brainstorming conversation to markdown file"
    )
    parser.add_argument("--topic", required=True, help="Topic name for the conversation")
    parser.add_argument("--user", required=True, help="User's input/question")
    parser.add_argument("--assistant", required=True, help="Claude's response")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be written without writing",
    )

    args = parser.parse_args()

    if args.dry_run:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = format_entry(timestamp, args.user, args.assistant)
        print("Would write to: ./argument/{}.qmd".format(args.topic))
        print("---")
        print(entry)
    else:
        filepath = save_conversation(args.topic, args.user, args.assistant)
        print(f"Saved to: {filepath}")


if __name__ == "__main__":
    main()
