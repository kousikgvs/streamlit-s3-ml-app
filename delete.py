"""
AWS Full Account Cleanup Script
Deletes: EC2 instances, key pairs, security groups, S3 buckets (all objects),
         IAM users (access keys, login profiles, group memberships, policies),
         across ALL regions.

Run:  python delete.py
"""

import boto3
from botocore.exceptions import ClientError

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_all_regions():
    ec2 = boto3.client("ec2", region_name="us-east-1")
    return [r["RegionName"] for r in ec2.describe_regions(AllRegions=False)["Regions"]]


# ── EC2 cleanup ────────────────────────────────────────────────────────────────

_SKIP_CODES = {"AuthFailure", "UnauthorizedAccess", "OptInRequired",
               "InvalidClientTokenId", "AccessDeniedException"}


def _is_skippable(e):
    return e.response["Error"]["Code"] in _SKIP_CODES


def terminate_ec2_instances(region):
    ec2 = boto3.client("ec2", region_name=region)
    try:
        paginator = ec2.get_paginator("describe_instances")
        instance_ids = []
        for page in paginator.paginate(Filters=[{"Name": "instance-state-name",
                                                  "Values": ["pending", "running", "stopping",
                                                             "stopped"]}]):
            for reservation in page["Reservations"]:
                for inst in reservation["Instances"]:
                    instance_ids.append(inst["InstanceId"])
        if instance_ids:
            ec2.terminate_instances(InstanceIds=instance_ids)
            print(f"  [{region}] Terminated EC2 instances: {instance_ids}")
        else:
            print(f"  [{region}] No EC2 instances found.")
    except ClientError as e:
        if _is_skippable(e):
            print(f"  [{region}] Skipped (region not enabled / auth failure).")
        else:
            raise


def delete_key_pairs(region):
    ec2 = boto3.client("ec2", region_name=region)
    try:
        pairs = ec2.describe_key_pairs()["KeyPairs"]
        for kp in pairs:
            ec2.delete_key_pair(KeyPairId=kp["KeyPairId"])
            print(f"  [{region}] Deleted key pair: {kp['KeyName']}")
        if not pairs:
            print(f"  [{region}] No key pairs found.")
    except ClientError as e:
        if _is_skippable(e):
            return  # already reported in terminate step
        raise


def delete_security_groups(region):
    """Delete non-default security groups."""
    ec2 = boto3.client("ec2", region_name=region)
    try:
        groups = ec2.describe_security_groups()["SecurityGroups"]
    except ClientError as e:
        if _is_skippable(e):
            return
        raise
    for sg in groups:
        if sg["GroupName"] == "default":
            continue
        try:
            ec2.delete_security_group(GroupId=sg["GroupId"])
            print(f"  [{region}] Deleted security group: {sg['GroupName']} ({sg['GroupId']})")
        except ClientError as e:
            print(f"  [{region}] Could not delete SG {sg['GroupId']}: {e.response['Error']['Code']}")


# ── S3 cleanup ─────────────────────────────────────────────────────────────────

def empty_and_delete_bucket(s3, bucket_name):
    """Delete all objects/versions then delete the bucket."""
    bucket = boto3.resource("s3").Bucket(bucket_name)

    # Delete all object versions (handles versioned buckets)
    bucket.object_versions.delete()

    # Delete any remaining objects (non-versioned)
    bucket.objects.delete()

    s3.delete_bucket(Bucket=bucket_name)
    print(f"  Deleted S3 bucket: {bucket_name}")


def delete_all_s3_buckets():
    s3 = boto3.client("s3", region_name="us-east-1")
    buckets = s3.list_buckets().get("Buckets", [])
    if not buckets:
        print("  No S3 buckets found.")
        return
    for bucket in buckets:
        try:
            empty_and_delete_bucket(s3, bucket["Name"])
        except ClientError as e:
            print(f"  Could not delete bucket {bucket['Name']}: {e.response['Error']['Code']}")


# ── IAM cleanup ────────────────────────────────────────────────────────────────

def delete_iam_users():
    iam = boto3.client("iam")
    paginator = iam.get_paginator("list_users")
    for page in paginator.paginate():
        for user in page["Users"]:
            username = user["UserName"]

            # Delete access keys
            for key in iam.list_access_keys(UserName=username)["AccessKeyMetadata"]:
                iam.delete_access_key(UserName=username, AccessKeyId=key["AccessKeyId"])

            # Delete login profile (console password)
            try:
                iam.delete_login_profile(UserName=username)
            except ClientError as e:
                if e.response["Error"]["Code"] != "NoSuchEntity":
                    raise

            # Detach managed policies
            for policy in iam.list_attached_user_policies(UserName=username)["AttachedPolicies"]:
                iam.detach_user_policy(UserName=username, PolicyArn=policy["PolicyArn"])

            # Delete inline policies
            for policy_name in iam.list_user_policies(UserName=username)["PolicyNames"]:
                iam.delete_user_policy(UserName=username, PolicyName=policy_name)

            # Remove from groups
            for group in iam.list_groups_for_user(UserName=username)["Groups"]:
                iam.remove_user_from_group(UserName=username, GroupName=group["GroupName"])

            # Delete MFA devices
            for mfa in iam.list_mfa_devices(UserName=username)["MFADevices"]:
                iam.deactivate_mfa_device(UserName=username, SerialNumber=mfa["SerialNumber"])
                iam.delete_virtual_mfa_device(SerialNumber=mfa["SerialNumber"])

            # Delete signing certificates
            for cert in iam.list_signing_certificates(UserName=username)["Certificates"]:
                iam.delete_signing_certificate(UserName=username, CertificateId=cert["CertificateId"])

            # Delete SSH public keys
            for key in iam.list_ssh_public_keys(UserName=username)["SSHPublicKeys"]:
                iam.delete_ssh_public_key(UserName=username, SSHPublicKeyId=key["SSHPublicKeyId"])

            iam.delete_user(UserName=username)
            print(f"  Deleted IAM user: {username}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  AWS FULL ACCOUNT CLEANUP")
    print("=" * 60)

    # S3 (global)
    print("\n[S3] Deleting all buckets...")
    delete_all_s3_buckets()

    # IAM users
    print("\n[IAM] Deleting all IAM users...")
    delete_iam_users()

    # EC2 per region
    print("\n[EC2] Cleaning up all regions...")
    regions = get_all_regions()
    for region in regions:
        print(f"\n  Region: {region}")
        terminate_ec2_instances(region)
        delete_key_pairs(region)
        delete_security_groups(region)

    print("\n" + "=" * 60)
    print("  Cleanup complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
