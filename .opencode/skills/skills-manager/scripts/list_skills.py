#!/usr/bin/env python3
"""Scan and list all installed skills with their metadata."""

import os
import yaml
from pathlib import Path


def parse_frontmatter(file_path):
    """Parse YAML frontmatter from a markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if not content.startswith('---'):
        return None

    # Find the end of frontmatter
    end_idx = content.find('---', 3)
    if end_idx == -1:
        return None

    frontmatter_text = content[3:end_idx]
    try:
        return yaml.safe_load(frontmatter_text)
    except yaml.YAMLError:
        return None


def get_skills_info(skills_dir):
    """Get information about all skills in the directory."""
    skills = []

    for skill_path in Path(skills_dir).iterdir():
        if not skill_path.is_dir() or skill_path.name.startswith('.'):
            continue

        skill_md = skill_path / 'SKILL.md'
        if not skill_md.exists():
            continue

        frontmatter = parse_frontmatter(skill_md)
        if not frontmatter:
            continue

        skills.append({
            'name': frontmatter.get('name', skill_path.name),
            'description': frontmatter.get('description', 'No description'),
            'directory': skill_path.name,
            'has_scripts': (skill_path / 'scripts').exists() and any((skill_path / 'scripts').iterdir()),
            'has_references': (skill_path / 'references').exists() and any((skill_path / 'references').iterdir()),
            'has_assets': (skill_path / 'assets').exists() and any((skill_path / 'assets').iterdir()),
        })

    return sorted(skills, key=lambda x: x['name'].lower())


def main():
    """Main function to output skills information."""
    skills_dir = Path.home() / '.claude' / 'skills'

    if not skills_dir.exists():
        print(f"# Skills directory not found: {skills_dir}")
        return

    skills = get_skills_info(skills_dir)

    if not skills:
        print("# No skills found")
        return

    print("# Available Skills\n")
    print(f"**Total**: {len(skills)} skills\n")

    for i, skill in enumerate(skills, 1):
        print(f"## {i}. {skill['name']}")
        print(f"**Directory**: `{skill['directory']}`\n")
        print(f"**Description**: {skill['description']}\n")

        # Show resources
        resources = []
        if skill['has_scripts']:
            resources.append("📜 Scripts")
        if skill['has_references']:
            resources.append("📚 References")
        if skill['has_assets']:
            resources.append("🎨 Assets")

        if resources:
            print(f"**Resources**: {', '.join(resources)}")

        print()


if __name__ == '__main__':
    main()
