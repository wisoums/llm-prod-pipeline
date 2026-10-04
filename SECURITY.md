# Security

This is a public repository. Do not commit secrets, credentials, private data, restricted datasets, or large model artifacts.

## Secrets

Keep tokens and credentials in environment variables or a local `.env` file. `.env` is ignored by Git. The committed `.env.example` must contain placeholders only.

Never commit:

- Hugging Face access tokens
- cloud/API credentials
- MLflow credentials
- private dataset contents
- model checkpoints or adapter weights that cannot legally be redistributed

If a secret is accidentally committed, revoke/rotate it immediately and remove it from Git history. Deleting the file in a later commit is not sufficient.

## Data and model licensing

Before adding a dataset or model artifact, verify its license and redistribution terms. Track source, license, version/revision, and preprocessing details in repository metadata rather than committing restricted raw content.

## Reporting vulnerabilities

Please open a private GitHub security advisory for security vulnerabilities rather than a public issue.
