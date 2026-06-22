import "@testing-library/jest-dom";
import { vi } from "vitest";

// jsdom does not implement these layout methods; framer-motion calls them during animations.
window.scrollTo = vi.fn();
Element.prototype.scrollIntoView = vi.fn();
