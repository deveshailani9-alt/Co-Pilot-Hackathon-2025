def main():
    import argparse

    parser = argparse.ArgumentParser(description='My Python Project CLI')
    parser.add_argument('command', type=str, help='The command to execute')
    parser.add_argument('--option', type=str, help='An optional argument for the command')

    args = parser.parse_args()

    if args.command == 'run':
        print('Running the application...')
        # Here you would call the main logic of your application
    elif args.command == 'test':
        print('Running tests...')
        # Here you would trigger your test suite
    else:
        print(f'Unknown command: {args.command}')

if __name__ == '__main__':
    main()