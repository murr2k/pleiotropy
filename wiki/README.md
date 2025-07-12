# Pleiotropy Wiki

This directory contains the wiki documentation for the Genomic Pleiotropy Cryptanalysis project.

## 📝 Wiki Structure

```
wiki/
├── Home.md                 # Main landing page
├── Current-Status.md       # System health and metrics
├── Roadmap.md             # Development timeline
├── Installation-Guide.md   # Setup instructions
├── Architecture.md         # Technical design
├── API-Reference.md        # API documentation
├── Algorithm-Details.md    # Mathematical foundations
├── Contributing.md         # Contribution guidelines
├── FAQ.md                 # Frequently asked questions
├── _Sidebar.md            # Navigation sidebar
└── README.md              # This file
```

## 🚀 Viewing the Wiki

The wiki is best viewed on GitHub at:
https://github.com/murr2k/pleiotropy/wiki

## ✏️ Contributing to the Wiki

### Making Changes

1. **Clone the wiki separately**:
   ```bash
   git clone https://github.com/murr2k/pleiotropy.wiki.git
   cd pleiotropy.wiki
   ```

2. **Create a new branch**:
   ```bash
   git checkout -b update-docs
   ```

3. **Make your changes**:
   - Follow Markdown best practices
   - Use clear headings
   - Include code examples
   - Add diagrams where helpful

4. **Commit and push**:
   ```bash
   git add .
   git commit -m "docs: update installation guide"
   git push origin update-docs
   ```

5. **Create a pull request** on the main repository

### Markdown Guidelines

- Use GitHub Flavored Markdown
- Include a table of contents for long pages
- Use code blocks with language hints
- Add alt text to images
- Link to other wiki pages using `[Page Name](Page-Name)`

### Page Naming

- Use Title Case for page names
- Replace spaces with hyphens
- Keep names concise but descriptive
- Examples: `Installation-Guide`, `API-Reference`

## 📊 Updating Status Information

The `Current-Status.md` page should be updated:
- After major deployments
- When adding new features
- Monthly for metrics
- When issues are resolved

## 🔄 Sync with Main Repository

The wiki content in this directory is periodically synced with the GitHub wiki. Changes should be made here first, then pushed to the wiki repository.

## 📚 Adding New Pages

When adding a new wiki page:

1. Create the `.md` file in this directory
2. Add it to the `_Sidebar.md` navigation
3. Link to it from relevant existing pages
4. Follow the existing format and style

## 🎨 Formatting Examples

### Code Blocks
```python
def example():
    return "Use syntax highlighting"
```

### Tables
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |

### Admonitions
> **Note**: Important information

> **Warning**: Critical warnings

> **Tip**: Helpful suggestions

## 🔗 Useful Links

- [GitHub Wiki Documentation](https://docs.github.com/en/communities/documenting-your-project-with-wikis)
- [Markdown Guide](https://www.markdownguide.org/)
- [Mermaid Diagrams](https://mermaid-js.github.io/mermaid/)

---

*For questions about wiki contributions, open an issue or ask in [Discussions](https://github.com/murr2k/pleiotropy/discussions).*