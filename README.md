# Daedalus

> The cooler PCGRL (Procedural Content Generation via Reinforcement Learning)

## 📋 Table of Contents

- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)
- [License](#license)

## 🔍 About

Daedalus is an advanced implementation of Procedural Content Generation via Reinforcement Learning (PCGRL). It provides enhanced capabilities for generating game content through reinforcement learning techniques.

## ✨ Features

- Improved PCGRL algorithms
- Modular architecture with pluggable components
- Customizable reward functions via critics
- Extensive testing framework

## 🚀 Installation

This project uses `uv` for package management.

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/daedalus.git
   cd daedalus
   ```

2. **Sync dependencies**
   ```bash
   uv sync
   ```

3. **Create a softlink for development**
   ```bash
   ln -s $(pwd) ./.venv/lib/python3.12/site-packages/$(basename $(pwd))
   ```

### Adding new packages

```bash
uv add <package-name>
```

## 💻 Usage

Run the main application:

```bash
python3 main.py
```

## 📁 Directory Structure

```
.
├── main.py           # Entry point of the application
├── agents/           # Reinforcement learning (RL) agents
├── models/           # Contains all RL/mdp models
├── critics/          # Reward functions for our agents
└── tests/            # Unit tests for all modules
```

## 🤝 Contributing

We welcome contributions to Daedalus! Please follow these guidelines:

1. **Coding Style**: Format your code using Black before committing:
   ```bash
   black .
   ```

2. **Commit Messages**: Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification:
   ```
   feat: add new feature
   fix: resolve bug
   docs: update documentation
   style: formatting changes
   refactor: code restructuring without changing functionality
   test: add or update tests
   chore: maintenance tasks
   ```

3. **Development Workflow**:
   - Work on a separate branch for each feature/fix
   - Create pull requests against the master branch
   - Write unit tests for any core functionality you add (in the `tests/` folder)
   - Ensure all tests pass before submitting your PR


---

<p align="center">
  Made with ❤️ by Team Lazarus
</p>
