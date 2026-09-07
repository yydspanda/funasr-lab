import { expect, test } from '@playwright/test';

test('mobile topic navigation leaves the article in the first viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/docs/training.html');
  const menu = page.locator('.docs-navigation');
  await expect(menu).not.toHaveAttribute('open', '');
  expect((await page.locator('.docs-title h1').boundingBox())!.y).toBeLessThan(300);
  await menu.locator(':scope > summary').press('Enter');
  await expect(menu).toHaveAttribute('open', '');
  await menu.locator('details').filter({ hasText: '训练与扩展' }).locator('summary').click();
  await page.locator('.docs-sidebar a[href="/docs/model-registration.html"]').click();
  await expect(page).toHaveURL(/model-registration.html$/);
  await page.setViewportSize({ width: 1440, height: 900 });
  await expect(page.locator('.docs-navigation')).toHaveAttribute('open', '');
  await expect(page.locator('.docs-sidebar summary').filter({ hasText: '训练与扩展' })).toBeVisible();
});

for (const width of [320, 768, 1440]) {
  test(`expanded documentation journey fits ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    for (const slug of ['installation', 'quickstart', 'python-api', 'model-zoo', 'training', 'model-registration', 'runtime-guide', 'service-api', 'security', 'kubernetes']) {
      await page.goto(`/docs/${slug}.html`, { waitUntil: 'domcontentloaded' });
      await expect(page.locator('.docs-article')).toBeVisible();
      expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth), slug).toBeLessThanOrEqual(1);
      await expect(page.locator('.docs-toc a').first()).toHaveAttribute('href', /^#/);
    }
    expect(errors).toEqual([]);
  });
}

test('SDK, training and Model Zoo are discoverable through search', async ({ page }) => {
  for (const [query, slug] of [['AutoModel', 'python-api'], ['Model Zoo', 'model-zoo'], ['fine-tuning', 'training']]) {
    await page.goto('/en/docs/');
    await page.locator('[data-doc-search] input').fill(query);
    await expect(page.locator(`.search-results a[href="/en/docs/${slug}.html"]`)).toBeVisible();
  }
});

for (const width of [320, 390, 768, 1440, 1920]) {
  test(`documentation and legacy layouts fit ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    for (const route of ['/docs/', '/en/docs/', '/docs/moss-transcribe-diarize.html', '/models.html', '/blog/', '/donors.html']) {
      await page.goto(route);
      await expect(page.locator('[data-primary-nav]')).toHaveCount(1);
      expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
    }
    expect(errors).toEqual([]);
  });
}

test('search is bilingual, keyboard accessible, and handles missing results', async ({ page }) => {
  for (const prefix of ['', '/en']) {
    await page.goto(`${prefix}/docs/`);
    const query = page.locator('[data-doc-search] input');
    await query.fill('MOSS');
    const first = page.locator('.search-results a').first();
    await expect(first).toBeVisible();
    await query.press('ArrowDown');
    await expect(first).toBeFocused();
    await expect(first).toHaveAttribute('href', new RegExp(`^${prefix}/`));
    await query.fill('zzzz-no-document-match-48291');
    await expect(page.locator('.search-results')).toContainText(prefix ? 'No matching documents' : '没有找到');
  }
});

test('source-derived code copies cleanly and legacy menu closes with Escape', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('/docs/moss-transcribe-diarize.html');
  const block = page.locator('.docs-article pre').first();
  const expected = await block.locator('code').innerText();
  await block.locator('button').click();
  expect((await page.evaluate(() => navigator.clipboard.readText())).trim()).toBe(expected.trim());
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/models.html');
  const menu = page.locator('[data-menu-toggle]');
  await menu.press('Enter');
  await expect(menu).toHaveAttribute('aria-expanded', 'true');
  await page.keyboard.press('Escape');
  await expect(menu).toHaveAttribute('aria-expanded', 'false');
});

test('public speech sample is playable and waveform has actual pixels', async ({ page }) => {
  await page.goto('/');
  await expect.poll(() => page.locator('audio').evaluate((audio: HTMLAudioElement) => audio.duration)).toBeGreaterThan(5);
  await page.locator('audio').evaluate((audio: HTMLAudioElement) => audio.play());
  await expect.poll(() => page.locator('audio').evaluate((audio: HTMLAudioElement) => audio.currentTime)).toBeGreaterThan(0);
  await page.locator('audio').evaluate((audio: HTMLAudioElement) => audio.pause());
  expect(await page.locator('.hero-image').evaluate((image: HTMLImageElement) => image.naturalWidth)).toBeGreaterThan(1000);
});
