# w-adebayo's AWS Automation Scripts
This directory is a personalized clone of the AWS_Rennie repository, curated for w-adebayo while preserving the original layout.

# Blog
Scripts from my Blog [Wale-Adebayo.com/blog](https://Wale-Adebayo.com/blog)

## Gitpod workspace helpers
The `gitpod/create-resources.sh` and `gitpod/destroy-resources.sh` scripts prepare a Gitpod workspace, install the CDK dependencies, and deploy or remove the `automation-rocks` stacks in `us-east-1` and `us-east-2`. Ensure your AWS credentials are available in the environment before running them (for example via `aws configure` or temporary session variables).

- Provision: `./gitpod/create-resources.sh`
- Cleanup: `./gitpod/destroy-resources.sh`

Set `SKIP_BOOTSTRAP=1` when provisioning if your account is already bootstrapped for CDK v1 in the target regions.
