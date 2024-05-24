#!/usr/bin/env python


import globus_sdk
from globus_sdk import TransferClient, TransferData, AuthClient
import os
from tqdm import tqdm

CLIENT_ID = '27704319-ba24-4676-9e99-96a6e0c68d65'
REDIRECT_URI = ''
TIMEOUT = 10

def main():
    # Set up a Native App authentication
    client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
    client.oauth2_start_flow(redirect_uri=REDIRECT_URI, requested_scopes="openid email profile urn:globus:auth:scope:transfer.api.globus.org:all")

    print('Please go to this URL and login: {0}'.format(client.oauth2_get_authorize_url()))

    auth_code = input('Please enter the code here: ').strip()

    # Exchange the authorization code for tokens
    token_response = client.oauth2_exchange_code_for_tokens(auth_code)
    transfer_tokens = token_response.by_resource_server['transfer.api.globus.org']

    # Create a Transfer client using the obtained tokens
    authorizer = globus_sdk.AccessTokenAuthorizer(transfer_tokens['access_token'])
    tc = globus_sdk.TransferClient(authorizer=authorizer)

    # Define source and destination endpoints
    source_endpoint_id = 'your_source_endpoint_id'
    destination_endpoint_id = 'your_destination_endpoint_id'

    # Create a transfer task
    transfer_data = TransferData(tc, source_endpoint_id, destination_endpoint_id)

    # Add files to the transfer task
    # For example, transferring all .hdf5 files from a specific folder on the source endpoint
    source_folder = '/path/to/source/folder'
    destination_folder = '/path/to/local/folder'

    # Ensure the local destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Add each HDF5 file to the transfer
    for file_path in tqdm(tc.operation_ls(source_endpoint_id, path=source_folder)):
        if file_path['type'] == 'file' and file_path['name'].endswith('.hdf5'):
            transfer_data.add_item(os.path.join(source_folder, file_path['name']), os.path.join(destination_folder, file_path['name']))

    # Submit the transfer
    transfer_result = tc.submit_transfer(transfer_data)
    print('Transfer task submitted, task ID:', transfer_result['task_id'])

    # Optionally, monitor the transfer status
    while not tc.task_wait(transfer_result['task_id'], timeout=TIMEOUT):
        print('Waiting for the transfer to complete...')

    print('Transfer completed!')

if __name__ == "__main__":
    main()