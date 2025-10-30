"""Example usage of the GitHub Events API client."""

import asyncio

from github_client import GitHubEventsClient


async def basic_example():
    """Basic example of fetching public events."""
    print("Example 1: Basic Usage")
    print("=" * 60)

    async with GitHubEventsClient() as client:
        response = await client.list_public_events(per_page=10)

        print(f"Fetched {len(response.events)} events")
        print(f"ETag: {response.etag}")
        print(f"Recommended poll interval: {response.poll_interval} seconds\n")

        for event in response.events[:5]:
            print(f"- {event.type}")
            print(f"  Actor: {event.actor.login}")
            print(f"  Repo: {event.repo.name}")
            print(f"  Time: {event.created_at}")
            print()


async def etag_caching_example():
    """Example demonstrating ETag caching for efficient polling."""
    print("\nExample 2: ETag Caching")
    print("=" * 60)

    async with GitHubEventsClient() as client:
        # First request
        print("Making initial request...")
        response1 = await client.list_public_events(per_page=5)
        etag = response1.etag
        poll_interval = response1.poll_interval

        print(f"Received {len(response1.events)} events")
        print(f"ETag: {etag}")
        print(f"Poll interval: {poll_interval} seconds")

        # Immediate second request with ETag (likely 304 Not Modified)
        print(f"\nMaking second request with ETag...")
        response2 = await client.list_public_events(etag=etag, per_page=5)

        if not response2.events:
            print("✓ No new events (304 Not Modified)")
            print("  Your rate limit was not affected!")
        else:
            print(f"✓ Received {len(response2.events)} new events")
            print(f"  New ETag: {response2.etag}")


async def polling_example():
    """Example of polling for new events with proper rate limiting."""
    print("\nExample 3: Polling with Rate Limiting")
    print("=" * 60)

    async with GitHubEventsClient() as client:
        etag = None
        poll_count = 0
        max_polls = 3

        while poll_count < max_polls:
            print(f"\nPoll #{poll_count + 1}")
            print("-" * 40)

            response = await client.list_public_events(
                per_page=5,
                etag=etag,
            )

            if response.events:
                print(f"✓ Received {len(response.events)} events")
                for event in response.events[:3]:
                    print(f"  - {event.type} by {event.actor.login}")

                # Update ETag for next poll
                etag = response.etag
            else:
                print("✓ No new events (304 Not Modified)")

            poll_count += 1

            # Respect the poll interval (or wait at least 5 seconds)
            wait_time = response.poll_interval or 5
            if poll_count < max_polls:
                print(f"\nWaiting {wait_time} seconds before next poll...")
                await asyncio.sleep(wait_time)


async def error_handling_example():
    """Example demonstrating error handling."""
    print("\nExample 4: Error Handling")
    print("=" * 60)

    from github_client import ForbiddenError, GitHubAPIError, ServiceUnavailableError

    async with GitHubEventsClient() as client:
        try:
            response = await client.list_public_events()
            print(f"✓ Successfully fetched {len(response.events)} events")

        except ForbiddenError as e:
            print(f"✗ Access forbidden: {e.message}")
            print("  You may have exceeded the rate limit")

        except ServiceUnavailableError as e:
            print(f"✗ Service unavailable: {e.message}")
            print("  Try again later")

        except GitHubAPIError as e:
            print(f"✗ API error: {e.message}")
            print(f"  Status code: {e.status_code}")


async def pagination_example():
    """Example demonstrating pagination."""
    print("\nExample 5: Pagination")
    print("=" * 60)

    async with GitHubEventsClient() as client:
        total_events = []

        # Fetch first 3 pages
        for page in range(1, 4):
            print(f"Fetching page {page}...")
            response = await client.list_public_events(
                per_page=10,
                page=page,
            )
            total_events.extend(response.events)
            print(f"  ✓ Fetched {len(response.events)} events")

        print(f"\nTotal events fetched: {len(total_events)}")


async def main():
    """Run all examples."""
    await basic_example()
    await etag_caching_example()
    await polling_example()
    await error_handling_example()
    await pagination_example()


if __name__ == "__main__":
    asyncio.run(main())
